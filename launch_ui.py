#!/usr/bin/env python3
"""
Sentient Web UI Launcher
Quick launcher script for the Streamlit web interface
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = ['streamlit', 'plotly', 'pandas']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    
    return True

def launch_ui():
    """Launch the Streamlit web UI"""
    
    # Change to the script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    print("✅ Dependencies OK")
    
    # Launch Streamlit
    print("🚀 Launching Sentient Web UI...")
    print("📱 The interface will open in your browser automatically")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "ui/app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--theme.base", "light",
            "--theme.primaryColor", "#667eea",
            "--theme.backgroundColor", "#ffffff",
            "--theme.secondaryBackgroundColor", "#f0f2f6"
        ])
    except KeyboardInterrupt:
        print("\n👋 Sentient UI stopped")
    except Exception as e:
        print(f"❌ Error launching UI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    launch_ui()