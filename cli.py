#!/usr/bin/env python3
"""
Sentient CLI - Command Line Interface for Consciousness AI
Interactive chat interface with the advanced consciousness system
"""

import argparse
import sys
import time
import json
from datetime import datetime
from typing import Optional

try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False
    # Fallback color definitions
    class Fore:
        RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = RESET = ""
    class Style:
        BRIGHT = DIM = RESET_ALL = ""

from consciousness_core import ConsciousnessAI, ProcessingMode

class SentientCLI:
    def __init__(self, mode: str = "consciousness", save_history: bool = True):
        """Initialize Sentient CLI"""
        
        # Initialize the AI
        self.ai = ConsciousnessAI(consciousness_enabled=True)
        self.mode = ProcessingMode(mode)
        self.save_history = save_history
        self.conversation_history = []
        
        # Display settings
        self.show_metrics = False
        self.show_timing = False
        
        self.print_banner()
        self.print_mode_info()
    
    def print_banner(self):
        """Print the CLI banner"""
        if COLORS_AVAILABLE:
            print(f"\n{Fore.CYAN}{Style.BRIGHT}╔══════════════════════════════════════════════════════════════╗")
            print(f"║  {Fore.MAGENTA}🧠 SENTIENT CONSCIOUSNESS AI - COMMAND LINE INTERFACE{Fore.CYAN}  ║")
            print(f"║  {Fore.WHITE}Advanced AI with Self-Awareness and Ethical Reasoning{Fore.CYAN}    ║")
            print(f"╚══════════════════════════════════════════════════════════════╝{Style.RESET_ALL}")
        else:
            print("\n" + "="*64)
            print("  🧠 SENTIENT CONSCIOUSNESS AI - COMMAND LINE INTERFACE")
            print("  Advanced AI with Self-Awareness and Ethical Reasoning")
            print("="*64)
        print()
    
    def print_mode_info(self):
        """Print current processing mode information"""
        mode_descriptions = {
            ProcessingMode.STANDARD: "Basic AI processing without consciousness enhancement",
            ProcessingMode.CONSCIOUSNESS: "Full consciousness-enhanced processing with self-awareness",
            ProcessingMode.CREATIVE: "Creative and innovative response generation",
            ProcessingMode.ETHICAL: "Ethically-aware processing with enhanced safety measures"
        }
        
        if COLORS_AVAILABLE:
            print(f"{Fore.YELLOW}Current Mode: {Fore.GREEN}{Style.BRIGHT}{self.mode.value.upper()}{Style.RESET_ALL}")
            print(f"{Fore.BLUE}Description: {Style.DIM}{mode_descriptions[self.mode]}{Style.RESET_ALL}")
        else:
            print(f"Current Mode: {self.mode.value.upper()}")
            print(f"Description: {mode_descriptions[self.mode]}")
        print()
    
    def print_help(self):
        """Print help information"""
        help_text = """
Available Commands:
  /help          - Show this help message
  /mode <mode>   - Change processing mode (standard, consciousness, creative, ethical)
  /metrics       - Toggle consciousness metrics display
  /timing        - Toggle processing time display
  /status        - Show system status
  /history       - Show conversation history
  /save [file]   - Save conversation to file
  /clear         - Clear conversation history
  /quit or /exit - Exit the CLI

Processing Modes:
  standard       - Basic AI processing
  consciousness  - Full consciousness enhancement
  creative       - Creative and innovative responses
  ethical        - Enhanced ethical reasoning

Tips:
  - Type naturally to have a conversation
  - Use /mode to switch between different AI personalities
  - Enable /metrics to see consciousness levels
  - All conversations are automatically saved (if enabled)
"""
        if COLORS_AVAILABLE:
            print(f"{Fore.CYAN}{help_text}{Style.RESET_ALL}")
        else:
            print(help_text)
    
    def process_command(self, user_input: str) -> bool:
        """Process CLI commands. Returns False if should exit."""
        
        if user_input.startswith('/'):
            parts = user_input[1:].split()
            command = parts[0].lower()
            args = parts[1:] if len(parts) > 1 else []
            
            if command in ['quit', 'exit', 'q']:
                return False
            
            elif command == 'help':
                self.print_help()
            
            elif command == 'mode':
                if args:
                    try:
                        new_mode = ProcessingMode(args[0])
                        self.mode = new_mode
                        self.print_mode_info()
                    except ValueError:
                        self.print_error(f"Invalid mode: {args[0]}")
                        self.print_info("Available modes: " + ", ".join([m.value for m in ProcessingMode]))
                else:
                    self.print_info(f"Current mode: {self.mode.value}")
            
            elif command == 'metrics':
                self.show_metrics = not self.show_metrics
                status = "enabled" if self.show_metrics else "disabled"
                self.print_info(f"Consciousness metrics display {status}")
            
            elif command == 'timing':
                self.show_timing = not self.show_timing
                status = "enabled" if self.show_timing else "disabled"
                self.print_info(f"Processing time display {status}")
            
            elif command == 'status':
                self.show_status()
            
            elif command == 'history':
                self.show_history()
            
            elif command == 'save':
                filename = args[0] if args else f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                self.save_conversation(filename)
            
            elif command == 'clear':
                self.conversation_history.clear()
                self.print_info("Conversation history cleared")
            
            else:
                self.print_error(f"Unknown command: /{command}")
                self.print_info("Type /help for available commands")
        
        return True
    
    def generate_response(self, prompt: str):
        """Generate and display AI response"""
        
        # Show user input
        if COLORS_AVAILABLE:
            print(f"{Fore.GREEN}You: {Style.RESET_ALL}{prompt}")
        else:
            print(f"You: {prompt}")
        
        # Generate response
        try:
            result = self.ai.generate(prompt, mode=self.mode)
            
            # Store in history
            self.conversation_history.append({
                'timestamp': datetime.now().isoformat(),
                'user_input': prompt,
                'ai_response': result.text,
                'mode': self.mode.value,
                'metrics': {
                    'consciousness': result.consciousness_metrics.overall_consciousness,
                    'confidence': result.confidence,
                    'processing_time': result.processing_time
                }
            })
            
            # Display response
            if COLORS_AVAILABLE:
                print(f"\n{Fore.MAGENTA}Sentient: {Style.RESET_ALL}{result.text}")
            else:
                print(f"\nSentient: {result.text}")
            
            # Show metrics if enabled
            if self.show_metrics:
                self.display_metrics(result)
            
            # Show timing if enabled
            if self.show_timing:
                if COLORS_AVAILABLE:
                    print(f"{Style.DIM}Processing time: {result.processing_time:.3f}s{Style.RESET_ALL}")
                else:
                    print(f"Processing time: {result.processing_time:.3f}s")
            
        except Exception as e:
            self.print_error(f"Error generating response: {str(e)}")
        
        print()  # Extra newline for spacing
    
    def display_metrics(self, result):
        """Display consciousness metrics"""
        metrics = result.consciousness_metrics
        
        if COLORS_AVAILABLE:
            print(f"\n{Style.DIM}Consciousness Metrics:")
            print(f"  Self-Awareness: {Fore.CYAN}{metrics.self_awareness:.1%}{Style.RESET_ALL}")
            print(f"  Cognitive Integration: {Fore.BLUE}{metrics.cognitive_integration:.1%}{Style.RESET_ALL}")
            print(f"  Ethical Reasoning: {Fore.GREEN}{metrics.ethical_reasoning:.1%}{Style.RESET_ALL}")
            print(f"  Overall Consciousness: {Fore.YELLOW}{metrics.overall_consciousness:.1%}{Style.RESET_ALL}")
            print(f"  Confidence: {Fore.MAGENTA}{result.confidence:.1%}{Style.RESET_ALL}")
        else:
            print("\nConsciousness Metrics:")
            print(f"  Self-Awareness: {metrics.self_awareness:.1%}")
            print(f"  Cognitive Integration: {metrics.cognitive_integration:.1%}")
            print(f"  Ethical Reasoning: {metrics.ethical_reasoning:.1%}")
            print(f"  Overall Consciousness: {metrics.overall_consciousness:.1%}")
            print(f"  Confidence: {result.confidence:.1%}")
    
    def show_status(self):
        """Show system status"""
        stats = self.ai.get_stats()
        status = self.ai.consciousness.get_system_status()
        
        if COLORS_AVAILABLE:
            print(f"\n{Fore.CYAN}System Status:{Style.RESET_ALL}")
            print(f"  Consciousness Level: {Fore.YELLOW}{status['consciousness_level']}{Style.RESET_ALL}")
            print(f"  Ethics Enabled: {Fore.GREEN if status['ethics_enabled'] else Fore.RED}{status['ethics_enabled']}{Style.RESET_ALL}")
            print(f"  Total Generations: {Fore.BLUE}{stats['total_generations']}{Style.RESET_ALL}")
            if stats['total_generations'] > 0:
                print(f"  Avg Consciousness: {Fore.MAGENTA}{stats['avg_consciousness_level']:.1%}{Style.RESET_ALL}")
                print(f"  Avg Confidence: {Fore.CYAN}{stats['avg_confidence']:.1%}{Style.RESET_ALL}")
        else:
            print("\nSystem Status:")
            print(f"  Consciousness Level: {status['consciousness_level']}")
            print(f"  Ethics Enabled: {status['ethics_enabled']}")
            print(f"  Total Generations: {stats['total_generations']}")
            if stats['total_generations'] > 0:
                print(f"  Avg Consciousness: {stats['avg_consciousness_level']:.1%}")
                print(f"  Avg Confidence: {stats['avg_confidence']:.1%}")
    
    def show_history(self):
        """Show conversation history"""
        if not self.conversation_history:
            self.print_info("No conversation history")
            return
        
        if COLORS_AVAILABLE:
            print(f"\n{Fore.CYAN}Conversation History ({len(self.conversation_history)} messages):{Style.RESET_ALL}")
        else:
            print(f"\nConversation History ({len(self.conversation_history)} messages):")
        
        for i, entry in enumerate(self.conversation_history[-5:], 1):  # Show last 5
            timestamp = datetime.fromisoformat(entry['timestamp']).strftime('%H:%M:%S')
            if COLORS_AVAILABLE:
                print(f"\n{Style.DIM}[{timestamp}]{Style.RESET_ALL}")
                print(f"{Fore.GREEN}You: {Style.RESET_ALL}{entry['user_input'][:100]}...")
                print(f"{Fore.MAGENTA}Sentient: {Style.RESET_ALL}{entry['ai_response'][:100]}...")
            else:
                print(f"\n[{timestamp}]")
                print(f"You: {entry['user_input'][:100]}...")
                print(f"Sentient: {entry['ai_response'][:100]}...")
    
    def save_conversation(self, filename: str):
        """Save conversation to file"""
        try:
            with open(filename, 'w') as f:
                json.dump({
                    'cli_session': {
                        'start_time': datetime.now().isoformat(),
                        'total_messages': len(self.conversation_history),
                        'mode_used': self.mode.value
                    },
                    'conversation': self.conversation_history
                }, f, indent=2)
            
            self.print_success(f"Conversation saved to {filename}")
        except Exception as e:
            self.print_error(f"Failed to save conversation: {str(e)}")
    
    def print_info(self, message: str):
        """Print info message"""
        if COLORS_AVAILABLE:
            print(f"{Fore.BLUE}ℹ {message}{Style.RESET_ALL}")
        else:
            print(f"ℹ {message}")
    
    def print_success(self, message: str):
        """Print success message"""
        if COLORS_AVAILABLE:
            print(f"{Fore.GREEN}✓ {message}{Style.RESET_ALL}")
        else:
            print(f"✓ {message}")
    
    def print_error(self, message: str):
        """Print error message"""
        if COLORS_AVAILABLE:
            print(f"{Fore.RED}✗ {message}{Style.RESET_ALL}")
        else:
            print(f"✗ {message}")
    
    def run(self):
        """Run the interactive CLI"""
        
        self.print_info("Type your message or use /help for commands. Type /quit to exit.")
        print()
        
        try:
            while True:
                # Get user input
                if COLORS_AVAILABLE:
                    prompt = f"{Fore.GREEN}You: {Style.RESET_ALL}"
                else:
                    prompt = "You: "
                
                try:
                    user_input = input(prompt).strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n")
                    break
                
                if not user_input:
                    continue
                
                # Process commands
                if user_input.startswith('/'):
                    if not self.process_command(user_input):
                        break
                else:
                    # Generate AI response
                    self.generate_response(user_input)
        
        except KeyboardInterrupt:
            print("\n")
        
        finally:
            # Save conversation before exit
            if self.save_history and self.conversation_history:
                filename = f"cli_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                self.save_conversation(filename)
            
            self.print_info("Thank you for using Sentient CLI!")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Sentient Consciousness AI - Command Line Interface")
    parser.add_argument('--mode', '-m', 
                       choices=['standard', 'consciousness', 'creative', 'ethical'],
                       default='consciousness',
                       help='Processing mode (default: consciousness)')
    parser.add_argument('--no-save', action='store_true',
                       help='Disable automatic conversation saving')
    parser.add_argument('--metrics', action='store_true',
                       help='Show consciousness metrics by default')
    parser.add_argument('--timing', action='store_true',
                       help='Show processing time by default')
    
    args = parser.parse_args()
    
    # Create and run CLI
    cli = SentientCLI(
        mode=args.mode,
        save_history=not args.no_save
    )
    
    if args.metrics:
        cli.show_metrics = True
    if args.timing:
        cli.show_timing = True
    
    cli.run()

if __name__ == "__main__":
    main()