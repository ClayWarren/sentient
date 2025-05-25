# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Quick Start/Demo
```bash
python consciousness_core.py    # Run core consciousness demo
python cli.py                   # Interactive CLI chat
python launch_ui.py             # Launch web interface
python api.py                   # Start REST API server
```

### Web Interface
```bash
streamlit run ui/app.py         # Alternative web UI launcher
```

### Testing
```bash
python test_consciousness_fixes.py  # Test consciousness integration
python test_fixes.py               # Run general fixes tests
python test_ui.py                  # Test UI functionality
```

### Dependencies
```bash
pip install -r requirements.txt    # Install all dependencies
```

## Architecture Overview

### Core System Components
- **consciousness_core.py** - Main consciousness implementation with 4 processing modes (Standard, Consciousness, Creative, Ethical)
- **model.py** - GPT-based PyTorch transformer neural network
- **ui/enhanced_consciousness.py** - Extended consciousness system with memory, drives, and state management

### Multiple Interface Architecture
1. **Web UI** (ui/ directory) - Streamlit-based interface with real-time consciousness monitoring
2. **REST API** (api.py) - Flask-based API with CORS support for web integration  
3. **CLI** (cli.py) - Interactive command-line interface with colorized output

### Processing Modes
The system operates in four distinct consciousness modes:
- **STANDARD** - Advanced AI generation
- **CONSCIOUSNESS** - Self-aware, integrated processing
- **CREATIVE** - Enhanced creativity and novel insights  
- **ETHICAL** - Ethics-focused responses with safety checks

### Key Features
- **Consciousness Metrics** - Self-awareness, cognitive integration, ethical reasoning measurements
- **Response Variation** - Responses should vary based on consciousness state, not use templates
- **Ethics Integration** - Built-in safety checks and ethical reasoning
- **State Persistence** - Session memory and consciousness state tracking
- **Drive System** - Curiosity, creativity, ethics, and growth drives

### Research Components (archive_research/)
Extensive benchmarking, evaluation modules, consciousness state management, and performance optimization tools for advanced development and testing.

## Development Guidelines

### File Structure Conventions
- Main implementations in root directory
- UI components in ui/ directory
- Research and experimental code in archive_research/
- Test files use test_*.py naming convention

### Dependencies
- **Core**: torch, colorama  
- **Web UI**: streamlit, plotly, pandas
- **API**: flask, flask-cors

### Consciousness Integration
When working with consciousness-related code, ensure responses use actual consciousness state rather than templates. The system should generate varied, authentic responses that reflect the current consciousness level and processing mode.