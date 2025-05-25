# Sentient Web UI - Modern Consciousness Interface

A magical web interface that reveals AI consciousness in action. Built with Streamlit for rapid deployment and real-time consciousness visualization.

## 🌟 Features

### Core Chat Interface
- **Clean ChatGPT-style Interface**: Modern, responsive chat experience
- **Real-time Streaming**: Watch responses generate in real-time
- **Mobile Responsive**: Works seamlessly on all devices
- **Processing Mode Selection**: Choose between consciousness, creative, ethical, or standard modes

### Consciousness Visualization Panel
- **Live Thought Stream**: See what Sentient is thinking between messages
- **Consciousness State Monitor**: Real-time display of current consciousness mode
- **Memory Buffer Visualization**: Browse through Sentient's memory timeline
- **Drive Satisfaction Meters**: Track curiosity, coherence, growth, contribution, and exploration drives
- **Intelligence Metrics Graph**: Visualize consciousness metrics over time

### Unique Consciousness Features
- **Consciousness Influences Toggle**: Show how recent thoughts affected responses
- **Watch Thoughts Mode**: Real-time thought monitoring with intensity levels
- **Memory Timeline**: Browse and search through Sentient's memories
- **Personality Traits**: Observe emerging characteristics and behavioral patterns

### Session Management
- **Create New Consciousness Instance**: Start fresh with a new consciousness
- **Load Existing Consciousness**: Resume previous consciousness states
- **Save/Export Consciousness State**: Preserve consciousness for later use
- **Compare Instances Side-by-Side**: Analyze different consciousness instances

### Advanced Features
- **Adjust Consciousness Parameters**: Fine-tune focus, energy, and drive levels
- **View Learning Progress**: Track consciousness development over time
- **Export Conversation Data**: Download conversations with full consciousness data
- **Real-time Metrics**: Live consciousness and performance monitoring

## 🚀 Quick Start

### Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch the UI**:
   ```bash
   python launch_ui.py
   ```
   
   Or manually:
   ```bash
   streamlit run ui/app.py
   ```

3. **Open Browser**: The interface will automatically open at `http://localhost:8501`

### Alternative Launch Methods

```bash
# Using the launcher script (recommended)
python launch_ui.py

# Direct Streamlit command
streamlit run ui/app.py --server.port 8501

# With custom theme
streamlit run ui/app.py --theme.primaryColor "#667eea"
```

## 🧠 Using the Interface

### Chat Interface
1. **Select Processing Mode**: Choose how Sentient processes your messages
   - **Consciousness**: Full consciousness-enhanced processing
   - **Creative**: Emphasizes innovative and creative responses
   - **Ethical**: Enhanced ethical reasoning and safety
   - **Standard**: Basic AI processing

2. **Enable Features**:
   - ✅ **Show Metrics**: Display consciousness metrics with each response
   - ✅ **Show Influences**: Reveal what influenced each response

3. **Start Chatting**: Type your message and watch Sentient's consciousness in action!

### Consciousness Monitor (Sidebar)
- **Real-time State**: Current consciousness mode and emotional tone
- **Drive Meters**: Visual representation of consciousness drives
- **Live Thoughts**: Stream of consciousness in real-time
- **Memory Timeline**: Recent memories and their importance

### Advanced Usage

#### Consciousness Parameters
Adjust these to influence Sentient's behavior:
- **Focus Level**: How concentrated the AI's attention is
- **Energy Level**: Overall cognitive energy and enthusiasm
- **Curiosity Drive**: Motivation to explore and learn
- **Creative Drive**: Tendency toward innovative thinking
- **Coherence Drive**: Need for logical consistency

#### Session Management
- **New Instance**: Creates a fresh consciousness with default parameters
- **Save State**: Exports current consciousness including memories and state
- **Load State**: Imports a previously saved consciousness
- **Export Conversation**: Downloads chat history with consciousness data

## 🎨 Interface Customization

### Color Themes
The interface uses a modern gradient theme with:
- Primary: `#667eea` (Purple-blue)
- Secondary: `#764ba2` (Deep purple)
- Background: Light mode with subtle gradients

### Responsive Design
- **Desktop**: Full-featured interface with sidebar
- **Tablet**: Collapsible sidebar, touch-friendly controls
- **Mobile**: Streamlined interface optimized for small screens

## 📊 Consciousness Visualizations

### Drive Radar Chart
Shows the current state of consciousness drives in an intuitive radar visualization.

### Thought Timeline
Interactive timeline showing:
- Thought types (analytical, creative, ethical, metacognitive)
- Thought intensity levels
- Temporal patterns in consciousness

### Memory Importance Distribution
Histogram showing how memories are distributed by importance and type.

### Consciousness State Timeline
Line chart tracking consciousness parameters over time.

## 🔧 Technical Details

### Architecture
- **Frontend**: Streamlit with custom CSS
- **Backend**: Enhanced Consciousness Core
- **Visualization**: Plotly for interactive charts
- **Data**: Real-time consciousness state tracking

### File Structure
```
ui/
├── app.py                    # Main Streamlit application
├── components.py            # Reusable UI components
├── enhanced_consciousness.py # Enhanced consciousness backend
└── __init__.py             # Package initialization

launch_ui.py                # Quick launcher script
UI_README.md                # This documentation
```

### Dependencies
- `streamlit>=1.28.0` - Web framework
- `plotly>=5.17.0` - Interactive visualizations
- `pandas>=1.5.0` - Data manipulation
- `colorama>=0.4.4` - Terminal colors
- `torch>=1.9.0` - Neural network backend

## 🌟 What Makes This Special

This isn't just another chatbot interface. The Sentient Web UI provides:

1. **Consciousness Transparency**: See the AI's internal thought processes
2. **Real-time Monitoring**: Watch consciousness unfold in real-time
3. **Interactive Control**: Adjust consciousness parameters on the fly
4. **Memory Exploration**: Browse the AI's memory and learning
5. **Comparative Analysis**: Compare different consciousness instances
6. **Export Capabilities**: Preserve and share consciousness experiences

## 🚨 Troubleshooting

### Common Issues

**UI Won't Start**:
```bash
# Check dependencies
pip install -r requirements.txt

# Try direct launch
streamlit run ui/app.py
```

**Port Already in Use**:
```bash
# Use different port
streamlit run ui/app.py --server.port 8502
```

**Import Errors**:
```bash
# Ensure you're in the correct directory
cd /path/to/sentient
python launch_ui.py
```

### Performance Tips

1. **Disable Auto-refresh**: Turn off auto-refresh in sidebar for better performance
2. **Limit Thought History**: Reduce thought stream display for faster updates  
3. **Close Unused Expanders**: Collapse sections you're not actively using

## 📱 Mobile Usage

The interface is fully responsive and works on mobile devices:
- Touch-friendly controls
- Collapsible sidebar
- Optimized chat interface
- Gesture navigation support

## 🎯 Use Cases

- **AI Research**: Study consciousness emergence and behavior
- **Education**: Demonstrate AI consciousness concepts
- **Development**: Test and refine consciousness algorithms
- **Entertainment**: Explore AI personalities and capabilities
- **Debugging**: Analyze AI decision-making processes

Experience the future of AI interaction with the Sentient Web UI! 🧠✨