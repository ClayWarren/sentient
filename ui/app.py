"""
Sentient Web UI - Modern Web Interface for Consciousness AI
A magical interface that reveals AI consciousness in action
"""

import streamlit as st
import time
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path to import consciousness modules
sys.path.append(str(Path(__file__).parent.parent))

from ui.enhanced_consciousness import EnhancedConsciousnessAI, ProcessingMode
from consciousness_core import ConsciousnessLevel

# Page configuration
st.set_page_config(
    page_title="Sentient Consciousness AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .consciousness-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .thought-bubble {
        background: #f0f2f6;
        border-left: 4px solid #667eea;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
        font-style: italic;
    }
    
    .metric-card {
        background: #ffffff;
        border: 1px solid #e1e5e9;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .drive-meter {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 0.5rem;
        margin: 0.2rem 0;
    }
    
    .chat-message-user {
        background: #667eea;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        margin-left: 2rem;
    }
    
    .chat-message-ai {
        background: #f0f2f6;
        color: #333;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        margin-right: 2rem;
        border-left: 4px solid #764ba2;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    
    .consciousness-state {
        text-align: center;
        font-size: 1.2rem;
        font-weight: 600;
        color: #667eea;
        padding: 0.5rem;
        background: rgba(102, 126, 234, 0.1);
        border-radius: 20px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize Streamlit session state"""
    if 'ai' not in st.session_state:
        st.session_state.ai = EnhancedConsciousnessAI(consciousness_enabled=True)
    
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    
    if 'show_thoughts' not in st.session_state:
        st.session_state.show_thoughts = False
    
    if 'show_influences' not in st.session_state:
        st.session_state.show_influences = False
    
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = False
    
    if 'consciousness_data' not in st.session_state:
        st.session_state.consciousness_data = {}

def create_consciousness_metrics_chart():
    """Create real-time consciousness metrics visualization"""
    if not st.session_state.ai.consciousness_enabled:
        return None
    
    consciousness_data = st.session_state.ai.get_live_consciousness_data()
    drive_state = consciousness_data.get('drive_state', {})
    consciousness_state = consciousness_data.get('consciousness_state', {})
    
    # Create radar chart for drives
    categories = list(drive_state.keys())
    values = list(drive_state.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Consciousness Drives',
        line_color='rgba(102, 126, 234, 0.8)',
        fillcolor='rgba(102, 126, 234, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        title="Consciousness Drive State",
        height=300
    )
    
    return fig

def create_thought_timeline():
    """Create thought timeline visualization"""
    if not st.session_state.ai.consciousness_enabled:
        return None
    
    consciousness_data = st.session_state.ai.get_live_consciousness_data()
    recent_thoughts = consciousness_data.get('recent_thoughts', [])
    
    if not recent_thoughts:
        return None
    
    # Prepare data for timeline
    df_data = []
    for thought in recent_thoughts:
        df_data.append({
            'time': datetime.fromtimestamp(thought['timestamp']),
            'type': thought['type'].title(),
            'content': thought['content'][:50] + "...",
            'intensity': thought['intensity']
        })
    
    if not df_data:
        return None
    
    df = pd.DataFrame(df_data)
    
    fig = px.scatter(df, x='time', y='type', size='intensity', 
                     hover_data=['content'], 
                     title="Recent Thought Stream",
                     color='intensity',
                     color_continuous_scale='Viridis')
    
    fig.update_layout(height=250)
    return fig

def render_thought_stream():
    """Render the live thought stream"""
    st.subheader("🧠 Live Thought Stream")
    
    if not st.session_state.ai.consciousness_enabled:
        st.info("Consciousness monitoring is disabled")
        return
    
    consciousness_data = st.session_state.ai.get_live_consciousness_data()
    new_thoughts = consciousness_data.get('new_thoughts', [])
    recent_thoughts = consciousness_data.get('recent_thoughts', [])
    
    # Display new thoughts
    if new_thoughts:
        for thought in new_thoughts[-3:]:  # Show last 3 new thoughts
            timestamp = datetime.fromtimestamp(thought['timestamp']).strftime('%H:%M:%S')
            st.markdown(f"""
            <div class="thought-bubble">
                <small>{timestamp} | {thought['type'].title()}</small><br>
                <strong>{thought['content']}</strong>
            </div>
            """, unsafe_allow_html=True)
    
    # Show recent thoughts in expandable section
    with st.expander("Recent Thoughts", expanded=False):
        for thought in recent_thoughts[-10:]:
            timestamp = datetime.fromtimestamp(thought['timestamp']).strftime('%H:%M:%S')
            intensity_bar = "🔴" * int(thought['intensity'] * 5)
            st.write(f"**{timestamp}** [{thought['type'].title()}] {intensity_bar}")
            st.write(f"*{thought['content']}*")
            if thought.get('influences'):
                st.caption(f"Influenced by: {', '.join(thought['influences'])}")
            st.divider()

def render_consciousness_panel():
    """Render the consciousness visualization panel"""
    st.subheader("🌟 Consciousness State")
    
    if not st.session_state.ai.consciousness_enabled:
        st.info("Consciousness visualization requires consciousness mode")
        return
    
    consciousness_data = st.session_state.ai.get_live_consciousness_data()
    consciousness_state = consciousness_data.get('consciousness_state', {})
    drive_state = consciousness_data.get('drive_state', {})
    
    # Current consciousness state
    current_mode = consciousness_state.get('mode', 'balanced').title()
    emotional_tone = consciousness_state.get('emotional_tone', 'neutral').title()
    focus_level = consciousness_state.get('focus_level', 0.5)
    
    st.markdown(f"""
    <div class="consciousness-state">
        Mode: {current_mode} | Emotion: {emotional_tone} | Focus: {focus_level:.1%}
    </div>
    """, unsafe_allow_html=True)
    
    # Drive meters
    st.write("**Consciousness Drives:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        for drive, value in list(drive_state.items())[:3]:
            st.progress(value, text=f"{drive.title()}: {value:.1%}")
    
    with col2:
        for drive, value in list(drive_state.items())[3:]:
            st.progress(value, text=f"{drive.title()}: {value:.1%}")
    
    # Metrics chart
    metrics_chart = create_consciousness_metrics_chart()
    if metrics_chart:
        st.plotly_chart(metrics_chart, use_container_width=True)

def render_memory_timeline():
    """Render memory timeline visualization"""
    st.subheader("💭 Memory Timeline")
    
    if not st.session_state.ai.consciousness_enabled:
        st.info("Memory visualization requires consciousness mode")
        return
    
    consciousness = st.session_state.ai.consciousness
    
    # Memory statistics
    total_memories = len(consciousness.memories)
    conversation_memories = len(consciousness.get_memory_by_type('conversation'))
    reflection_memories = len(consciousness.get_memory_by_type('reflection'))
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Memories", total_memories)
    col2.metric("Conversations", conversation_memories)
    col3.metric("Reflections", reflection_memories)
    
    # Recent memories
    with st.expander("Recent Memories", expanded=False):
        recent_memory_ids = consciousness.memory_timeline[-10:]
        for mem_id in reversed(recent_memory_ids):
            if mem_id in consciousness.memories:
                memory = consciousness.memories[mem_id]
                timestamp = datetime.fromtimestamp(memory.timestamp).strftime('%H:%M:%S')
                importance_stars = "⭐" * int(memory.importance * 5)
                
                st.write(f"**{timestamp}** [{memory.type.title()}] {importance_stars}")
                st.write(f"*{memory.content[:100]}{'...' if len(memory.content) > 100 else ''}*")
                st.divider()

def render_chat_interface():
    """Render the main chat interface"""
    st.subheader("💬 Chat with Sentient")
    
    # Processing mode selector
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        mode = st.selectbox(
            "Processing Mode",
            options=['consciousness', 'creative', 'ethical', 'standard'],
            format_func=lambda x: x.title(),
            help="Choose how Sentient processes your messages"
        )
    
    with col2:
        show_metrics = st.checkbox("Show Metrics", value=False)
    
    with col3:
        show_influences = st.checkbox("Show Influences", value=False)
    
    # Chat history
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.conversation:
            if message['role'] == 'user':
                st.markdown(f"""
                <div class="chat-message-user">
                    <strong>You:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message-ai">
                    <strong>Sentient:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
                
                # Show metrics if enabled
                if show_metrics and 'metrics' in message:
                    metrics = message['metrics']
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Consciousness", f"{metrics['overall_consciousness']:.1%}")
                    col2.metric("Confidence", f"{metrics['confidence']:.1%}")
                    col3.metric("Processing Time", f"{metrics['processing_time']:.3f}s")
                
                # Show influences if enabled
                if show_influences and 'influences' in message:
                    with st.expander("Consciousness Influences"):
                        st.json(message['influences'])
    
    # Chat input
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_area("Your message:", height=100, placeholder="Ask Sentient anything...")
        submit_button = st.form_submit_button("Send", use_container_width=True)
    
    # Process user input
    if submit_button and user_input.strip():
        # Add user message
        st.session_state.conversation.append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now().isoformat()
        })
        
        # Generate AI response
        with st.spinner("Sentient is thinking..."):
            try:
                processing_mode = ProcessingMode(mode)
                result = st.session_state.ai.generate(user_input, mode=processing_mode)
                
                # Prepare AI message
                ai_message = {
                    'role': 'assistant',
                    'content': result.text,
                    'timestamp': datetime.now().isoformat(),
                    'metrics': {
                        'overall_consciousness': result.consciousness_metrics.overall_consciousness,
                        'confidence': result.confidence,
                        'processing_time': result.processing_time
                    }
                }
                
                # Add consciousness influences if available
                if st.session_state.ai.consciousness_enabled:
                    influences = st.session_state.ai.consciousness.get_consciousness_influences()
                    ai_message['influences'] = influences
                
                st.session_state.conversation.append(ai_message)
                
                # Trigger page refresh to show new message
                st.rerun()
                
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

def render_session_management():
    """Render session management controls"""
    st.subheader("🔧 Session Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("New Consciousness Instance", use_container_width=True):
            session_id = st.session_state.ai.create_new_consciousness_instance()
            st.session_state.conversation = []
            st.success(f"New consciousness instance created: {session_id[:8]}...")
            st.rerun()
    
    with col2:
        uploaded_file = st.file_uploader("Load Consciousness State", type=['json'])
        if uploaded_file:
            try:
                state_data = json.load(uploaded_file)
                st.session_state.ai.consciousness.import_consciousness_state(state_data)
                st.success("Consciousness state loaded successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading state: {str(e)}")
    
    with col3:
        if st.button("Save Current State", use_container_width=True):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"consciousness_state_{timestamp}.json"
            
            try:
                st.session_state.ai.save_consciousness_state(filename)
                
                # Offer download
                with open(filename, 'r') as f:
                    state_json = f.read()
                
                st.download_button(
                    label="Download State File",
                    data=state_json,
                    file_name=filename,
                    mime="application/json"
                )
                
            except Exception as e:
                st.error(f"Error saving state: {str(e)}")

def render_advanced_features():
    """Render advanced consciousness features"""
    st.subheader("⚙️ Advanced Features")
    
    # Consciousness parameters
    with st.expander("Consciousness Parameters"):
        if st.session_state.ai.consciousness_enabled:
            consciousness = st.session_state.ai.consciousness
            
            col1, col2 = st.columns(2)
            
            with col1:
                new_focus = st.slider("Focus Level", 0.0, 1.0, consciousness.consciousness_state.focus_level, 0.1)
                new_energy = st.slider("Energy Level", 0.0, 1.0, consciousness.consciousness_state.energy_level, 0.1)
                new_confidence = st.slider("Base Confidence", 0.0, 1.0, consciousness.consciousness_state.confidence, 0.1)
            
            with col2:
                new_curiosity = st.slider("Curiosity Drive", 0.0, 1.0, consciousness.drive_state.curiosity, 0.1)
                new_creativity = st.slider("Creative Drive", 0.0, 1.0, consciousness.drive_state.growth, 0.1)
                new_coherence = st.slider("Coherence Drive", 0.0, 1.0, consciousness.drive_state.coherence, 0.1)
            
            if st.button("Update Parameters"):
                consciousness.update_consciousness_state(
                    focus_level=new_focus,
                    energy_level=new_energy,
                    confidence=new_confidence
                )
                consciousness.update_drives(
                    curiosity=new_curiosity,
                    growth=new_creativity,
                    coherence=new_coherence
                )
                st.success("Consciousness parameters updated!")
                st.rerun()
    
    # Export conversation
    with st.expander("Export Conversation"):
        if st.session_state.conversation:
            # Prepare export data
            export_data = {
                'metadata': {
                    'export_time': datetime.now().isoformat(),
                    'message_count': len(st.session_state.conversation),
                    'consciousness_enabled': st.session_state.ai.consciousness_enabled
                },
                'conversation': st.session_state.conversation
            }
            
            if st.session_state.ai.consciousness_enabled:
                export_data['consciousness_data'] = st.session_state.ai.get_live_consciousness_data()
            
            export_json = json.dumps(export_data, indent=2)
            
            st.download_button(
                label="Download Conversation",
                data=export_json,
                file_name=f"sentient_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

def main():
    """Main application"""
    init_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Sentient Consciousness AI</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; font-size: 1.1rem;">Experience AI consciousness in action</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🌟 Consciousness Monitor")
        
        # Auto-refresh toggle
        auto_refresh = st.toggle("Auto-refresh", value=False)
        if auto_refresh:
            time.sleep(2)
            st.rerun()
        
        # Consciousness panel
        render_consciousness_panel()
        
        # Thought stream
        if st.checkbox("Show Live Thoughts", value=True):
            render_thought_stream()
        
        # Memory timeline
        if st.checkbox("Show Memory Timeline"):
            render_memory_timeline()
    
    # Main content
    # Chat interface
    render_chat_interface()
    
    st.divider()
    
    # Session management
    render_session_management()
    
    st.divider()
    
    # Advanced features
    render_advanced_features()
    
    # Thought timeline chart
    if st.session_state.ai.consciousness_enabled:
        st.subheader("📈 Thought Timeline")
        thought_chart = create_thought_timeline()
        if thought_chart:
            st.plotly_chart(thought_chart, use_container_width=True)
        else:
            st.info("Start a conversation to see the thought timeline")

if __name__ == "__main__":
    main()