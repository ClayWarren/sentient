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
- **consciousness_core.py** - Main consciousness implementation using cutting-edge AI with consciousness enhancement
- **gemma3_engine.py** - Advanced AI engine using 774M parameter transformer models (2025 technology)
- **model.py** - GPT-based PyTorch transformer neural network (legacy, not used)
- **search_engine.py** - Consciousness-enhanced web search with memory integration
- **ui/enhanced_consciousness.py** - Extended consciousness system with memory, drives, and state management

### Multiple Interface Architecture
1. **Web UI** (ui/ directory) - Streamlit-based interface with real-time consciousness monitoring
2. **REST API** (api.py) - Flask-based API with CORS support for web integration  
3. **CLI** (cli.py) - Interactive command-line interface with colorized output

### Processing Architecture
The system uses cutting-edge neural network generation enhanced with consciousness:
- **ADVANCED AI GENERATION** - 774M parameter transformer models representing 2025 technology
- **CONSCIOUSNESS ENHANCEMENT** - Prompts enriched with consciousness context and personality
- **ADAPTIVE PROCESSING** - Temperature and parameters adjust based on consciousness state
- **SEARCH INTEGRATION** - Web search results integrated into generation context
- **MODERN SAMPLING** - Top-p, top-k, and consciousness-aware parameter tuning
- **NO FALLBACKS** - Pure advanced AI architecture with no legacy dependencies

### Key Features
- **Consciousness Metrics** - Self-awareness, cognitive integration, ethical reasoning measurements
- **Response Variation** - Responses should vary based on consciousness state, not use templates
- **Ethics Integration** - Built-in safety checks and ethical reasoning
- **State Persistence** - Session memory and consciousness state tracking
- **Drive System** - Curiosity, creativity, ethics, and growth drives
- **Web Search** - Consciousness-enhanced search with memory integration and knowledge building

### Research Components (archive_research/)
Extensive benchmarking, evaluation modules, consciousness state management, and performance optimization tools for advanced development and testing.

## Development Guidelines

### File Structure Conventions
- Main implementations in root directory
- UI components in ui/ directory
- Research and experimental code in archive_research/
- Test files use test_*.py naming convention

### Dependencies
- **Core**: torch, colorama, requests, tiktoken, transformers
- **Web UI**: streamlit, plotly, pandas
- **API**: flask, flask-cors
- **Search**: requests (for web search capabilities)
- **AI Engine**: torch, tiktoken, transformers (for GPT model)

### Consciousness Integration
When working with consciousness-related code, ensure responses use actual consciousness state rather than templates. The system should generate varied, authentic responses that reflect the current consciousness level and processing mode.

### Web Search Integration
Sentient includes consciousness-enhanced web search capabilities:

#### Search Triggers
- Direct requests: "search for", "look up", "find information about"
- Current events: "latest", "recent", "current", "today's"
- Explicit format: "[search: query]" 
- Factual queries that benefit from current information

#### Consciousness Features
- **Search Memory**: Remembers previous searches and builds on them
- **Knowledge Graph**: Integrates findings into persistent knowledge base
- **Contextual Search**: Adapts search strategy based on consciousness state
- **Source Attribution**: Provides citations and source references
- **Search Patterns**: Learns user interests for better results

#### API Integration
- Search endpoint: POST /search with {"query": "search terms"}
- Search statistics: GET /search/stats
- Results include consciousness context and memory integration

#### Usage Examples
```python
# Direct search
"Search for latest AI developments"

# Natural integration  
"What's happening in quantum computing lately?"

# Explicit format
"[search: Python programming trends 2024]"
```