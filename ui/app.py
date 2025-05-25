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
import base64
import hashlib
import os
from typing import Dict, List, Any
import PyPDF2
from PIL import Image
import docx
import csv
from io import StringIO, BytesIO

# Add parent directory to path to import consciousness modules
sys.path.append(str(Path(__file__).parent.parent))

from ui.enhanced_consciousness import EnhancedConsciousnessAI
from consciousness_core import ConsciousnessLevel

# Page configuration
st.set_page_config(
    page_title="Sentient - The AI that remembers and thinks continuously",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design with new features
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
    
    .file-upload-zone {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        background: rgba(102, 126, 234, 0.05);
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    
    .file-upload-zone:hover {
        border-color: #764ba2;
        background: rgba(102, 126, 234, 0.1);
    }
    
    .uploaded-file {
        background: linear-gradient(90deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
        border-left: 4px solid #667eea;
    }
    
    .thought-ticker {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 8px 15px;
        border-radius: 20px;
        margin: 5px 0;
        font-size: 0.9rem;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    .brain-icon {
        animation: pulse-brain 1.5s infinite;
        font-size: 1.2rem;
    }
    
    @keyframes pulse-brain {
        0% { opacity: 0.7; }
        50% { opacity: 1; }
        100% { opacity: 0.7; }
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
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
    
    .voice-controls {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 50px;
        padding: 10px;
        text-align: center;
        margin: 10px 0;
    }
    
    .voice-button {
        background: rgba(255, 255, 255, 0.2);
        border: none;
        border-radius: 50%;
        width: 60px;
        height: 60px;
        color: white;
        font-size: 1.5rem;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .voice-button:hover {
        background: rgba(255, 255, 255, 0.3);
        transform: scale(1.1);
    }
    
    .memory-indicator {
        width: 100%;
        height: 4px;
        background: rgba(102, 126, 234, 0.2);
        border-radius: 2px;
        overflow: hidden;
        margin: 5px 0;
    }
    
    .memory-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 2px;
        transition: width 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def process_uploaded_file(uploaded_file) -> Dict[str, Any]:
    """Process uploaded file and extract content"""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        file_content = ""
        file_type = "unknown"
        
        # Generate file hash for deduplication
        file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
        
        # Process based on file type
        if file_extension == 'pdf':
            file_type = "PDF Document"
            pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.getvalue()))
            for page in pdf_reader.pages:
                file_content += page.extract_text() + "\n"
        
        elif file_extension == 'txt':
            file_type = "Text Document"
            file_content = str(uploaded_file.getvalue(), 'utf-8')
        
        elif file_extension == 'csv':
            file_type = "CSV Data"
            csv_content = StringIO(str(uploaded_file.getvalue(), 'utf-8'))
            csv_reader = csv.reader(csv_content)
            rows = list(csv_reader)
            file_content = f"CSV with {len(rows)} rows and {len(rows[0]) if rows else 0} columns:\n"
            file_content += "\n".join([",".join(row) for row in rows[:10]])  # Show first 10 rows
            if len(rows) > 10:
                file_content += f"\n... and {len(rows) - 10} more rows"
        
        elif file_extension == 'docx':
            file_type = "Word Document"
            doc = docx.Document(BytesIO(uploaded_file.getvalue()))
            for paragraph in doc.paragraphs:
                file_content += paragraph.text + "\n"
        
        elif file_extension in ['jpg', 'jpeg', 'png', 'gif', 'bmp']:
            file_type = "Image"
            image = Image.open(BytesIO(uploaded_file.getvalue()))
            file_content = f"Image: {image.format} format, {image.size[0]}x{image.size[1]} pixels"
            # In a real implementation, you might use OCR or image description AI
        
        else:
            file_type = "Unknown File"
            file_content = "File type not supported for content extraction"
        
        return {
            'name': uploaded_file.name,
            'type': file_type,
            'size': uploaded_file.size,
            'content': file_content[:5000],  # Limit content length
            'hash': file_hash,
            'timestamp': datetime.now().isoformat(),
            'memory_importance': min(1.0, len(file_content) / 10000)  # Calculate importance
        }
    
    except Exception as e:
        return {
            'name': uploaded_file.name,
            'type': "Error",
            'size': uploaded_file.size,
            'content': f"Error processing file: {str(e)}",
            'hash': hashlib.md5(uploaded_file.getvalue()).hexdigest(),
            'timestamp': datetime.now().isoformat(),
            'memory_importance': 0.1
        }

def add_file_to_memory(file_info: Dict[str, Any]):
    """Add processed file to Sentient's memory"""
    if st.session_state.ai.consciousness_enabled:
        consciousness = st.session_state.ai.consciousness
        
        # Create memory content
        memory_content = f"Uploaded file: {file_info['name']} ({file_info['type']})\n"
        memory_content += f"Content preview: {file_info['content'][:500]}..."
        
        # Add to consciousness memory
        memory_id = consciousness.add_memory(
            content=memory_content,
            memory_type="file_upload",
            importance=file_info['memory_importance'],
            metadata={
                'filename': file_info['name'],
                'file_type': file_info['type'],
                'file_size': file_info['size'],
                'file_hash': file_info['hash']
            }
        )
        
        # Store file info in session state
        st.session_state.file_memories[file_info['hash']] = {
            'memory_id': memory_id,
            'file_info': file_info
        }

def render_file_upload_zone():
    """Render the drag-and-drop file upload area"""
    st.markdown("""
    <div class="file-upload-zone">
        <h4>📁 Drag & Drop Files Here</h4>
        <p>Support: PDF, TXT, CSV, DOCX, Images (JPG, PNG, GIF)</p>
        <p>Files will be processed and added to Sentient's memory</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader (fallback for browsers without drag-drop)
    uploaded_files = st.file_uploader(
        "Or click to upload files",
        type=['pdf', 'txt', 'csv', 'docx', 'jpg', 'jpeg', 'png', 'gif', 'bmp'],
        accept_multiple_files=True,
        help="Upload files to add to Sentient's working memory"
    )
    
    if uploaded_files:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}...")
            
            # Check if file already exists
            file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
            if file_hash not in [f['hash'] for f in st.session_state.uploaded_files]:
                
                # Process file
                file_info = process_uploaded_file(uploaded_file)
                
                # Add to session state
                st.session_state.uploaded_files.append(file_info)
                
                # Add to memory
                add_file_to_memory(file_info)
                
                st.success(f"✅ Processed: {uploaded_file.name}")
            else:
                st.info(f"📄 File already uploaded: {uploaded_file.name}")
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        status_text.text("All files processed!")
        st.rerun()

def render_live_analytics_dashboard():
    """Render live analytics dashboard with interactive charts"""
    st.subheader("📊 Live Analytics Dashboard")
    
    if not st.session_state.ai.consciousness_enabled:
        st.info("Analytics require consciousness mode")
        return
    
    consciousness_data = st.session_state.ai.get_live_consciousness_data()
    
    # Real-time metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Thoughts per second calculation
        recent_thoughts = consciousness_data.get('recent_thoughts', [])
        current_time = time.time()
        last_10_seconds = current_time - 10
        recent_count = len([t for t in recent_thoughts if t['timestamp'] > last_10_seconds])
        thoughts_per_second = recent_count / 10
        
        st.metric("Thoughts/Second", f"{thoughts_per_second:.2f}", delta=f"{thoughts_per_second - 0.5:.2f}")
    
    with col2:
        # Memory utilization
        total_memories = len(st.session_state.ai.consciousness.memories)
        memory_utilization = min(total_memories / 100, 1.0)  # Assume 100 is full capacity
        st.metric("Memory Usage", f"{memory_utilization:.1%}", delta=f"{len(st.session_state.uploaded_files)} files")
    
    with col3:
        # Average consciousness level
        if recent_thoughts:
            avg_intensity = sum(t['intensity'] for t in recent_thoughts[-10:]) / min(10, len(recent_thoughts))
            st.metric("Avg Consciousness", f"{avg_intensity:.1%}", delta=f"{avg_intensity - 0.7:.1%}")
        else:
            st.metric("Avg Consciousness", "0%")
    
    with col4:
        # Conversation turns
        conversation_turns = len(st.session_state.conversation)
        st.metric("Conversation Turns", conversation_turns, delta=f"{conversation_turns % 10} recent")
    
    # Interactive charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Memory utilization pie chart
        if st.session_state.ai.consciousness_enabled:
            consciousness = st.session_state.ai.consciousness
            memory_types = {}
            
            for memory in consciousness.memories.values():
                mem_type = memory.type
                if mem_type not in memory_types:
                    memory_types[mem_type] = 0
                memory_types[mem_type] += 1
            
            if memory_types:
                fig_pie = px.pie(
                    values=list(memory_types.values()),
                    names=list(memory_types.keys()),
                    title="Memory Distribution by Type"
                )
                fig_pie.update_layout(height=300)
                st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Real-time thought intensity graph
        if recent_thoughts:
            df_intensity = pd.DataFrame([
                {
                    'time': datetime.fromtimestamp(t['timestamp']),
                    'intensity': t['intensity'],
                    'type': t['type']
                } for t in recent_thoughts[-20:]  # Last 20 thoughts
            ])
            
            fig_line = px.line(
                df_intensity, 
                x='time', 
                y='intensity',
                color='type',
                title="Thought Intensity Over Time"
            )
            fig_line.update_layout(height=300)
            st.plotly_chart(fig_line, use_container_width=True)
    
    # Personality trait evolution (simulated data for demo)
    with st.expander("🧬 Personality Evolution", expanded=False):
        # Create sample personality data
        personality_data = {
            'trait': ['Curiosity', 'Creativity', 'Empathy', 'Logic', 'Humor'],
            'current': [0.8, 0.6, 0.7, 0.9, 0.5],
            'previous': [0.7, 0.5, 0.6, 0.8, 0.4]
        }
        
        df_personality = pd.DataFrame(personality_data)
        fig_personality = px.bar(
            df_personality,
            x='trait',
            y=['current', 'previous'],
            title="Personality Trait Evolution",
            barmode='group'
        )
        st.plotly_chart(fig_personality, use_container_width=True)

def render_uploaded_files_sidebar():
    """Render uploaded files in sidebar with memory indicators"""
    if st.session_state.uploaded_files:
        st.subheader("📁 Uploaded Files")
        
        for file_info in st.session_state.uploaded_files:
            with st.expander(f"📄 {file_info['name']}", expanded=False):
                st.write(f"**Type:** {file_info['type']}")
                st.write(f"**Size:** {file_info['size']} bytes")
                st.write(f"**Uploaded:** {file_info['timestamp'][:19]}")
                
                # Memory importance indicator
                importance = file_info['memory_importance']
                st.markdown(f"""
                <div class="memory-indicator">
                    <div class="memory-fill" style="width: {importance * 100}%"></div>
                </div>
                <small>Memory Importance: {importance:.1%}</small>
                """, unsafe_allow_html=True)
                
                # Preview content
                if len(file_info['content']) > 100:
                    st.text_area("Content Preview:", 
                               value=file_info['content'][:200] + "...", 
                               height=100, 
                               disabled=True)
                
                # Remove file button
                if st.button(f"Remove {file_info['name']}", key=f"remove_{file_info['hash']}"):
                    st.session_state.uploaded_files = [
                        f for f in st.session_state.uploaded_files 
                        if f['hash'] != file_info['hash']
                    ]
                    if file_info['hash'] in st.session_state.file_memories:
                        del st.session_state.file_memories[file_info['hash']]
                    st.rerun()

def render_live_thought_ticker():
    """Render the live thought stream ticker at top of UI"""
    if st.session_state.show_live_thoughts and st.session_state.ai.consciousness_enabled:
        st.markdown("""
        <div style="position: sticky; top: 0; z-index: 999; background: rgba(255,255,255,0.95); padding: 10px; border-bottom: 1px solid #eee;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span class="brain-icon">🧠</span>
                <div id="thought-ticker" style="flex: 1; overflow: hidden;">
        """, unsafe_allow_html=True)
        
        # Get latest thoughts
        consciousness_data = st.session_state.ai.get_live_consciousness_data()
        new_thoughts = consciousness_data.get('new_thoughts', [])
        
        if new_thoughts:
            latest_thought = new_thoughts[-1]
            thought_type = latest_thought['type'].title()
            thought_content = latest_thought['content']
            intensity = latest_thought['intensity']
            
            # Calculate thought rate (thoughts per minute)
            recent_thoughts = consciousness_data.get('recent_thoughts', [])
            current_time = time.time()
            minute_ago = current_time - 60
            recent_count = len([t for t in recent_thoughts if t['timestamp'] > minute_ago])
            
            st.markdown(f"""
                <div class="thought-ticker">
                    <strong>[{thought_type}]</strong> {thought_content[:100]}...
                    <small style="float: right;">Rate: {recent_count}/min | Intensity: {'🔴' * int(intensity * 5)}</small>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="thought-ticker">
                    <em>Consciousness at rest... waiting for interaction</em>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div></div></div>", unsafe_allow_html=True)

def render_voice_controls():
    """Render voice mode controls"""
    st.markdown("""
    <div class="voice-controls">
        <h4 style="color: white; text-align: center; margin: 0;">🎤 Voice Mode</h4>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎤 Start Recording", key="start_recording", help="Click to start voice recording"):
            st.session_state.voice_recording = True
            st.rerun()
    
    with col2:
        if st.button("🔊 Read Aloud", key="text_to_speech", help="Read the last AI response aloud"):
            if st.session_state.conversation:
                last_ai_message = None
                for msg in reversed(st.session_state.conversation):
                    if msg['role'] == 'assistant':
                        last_ai_message = msg['content']
                        break
                
                if last_ai_message:
                    # Use JavaScript for text-to-speech
                    speech_js = f"""
                    <script>
                    const utterance = new SpeechSynthesisUtterance("{last_ai_message.replace('"', '')}");
                    utterance.rate = 0.8;
                    utterance.pitch = 1;
                    utterance.volume = 0.8;
                    speechSynthesis.speak(utterance);
                    </script>
                    """
                    st.markdown(speech_js, unsafe_allow_html=True)
                    st.success("🔊 Speaking...")
    
    with col3:
        voice_mode = st.checkbox("Continuous Mode", help="Auto-speak AI responses")
        if voice_mode:
            st.session_state.voice_mode_enabled = True
        else:
            st.session_state.voice_mode_enabled = False
    
    # Speech recognition interface
    if st.session_state.get('voice_recording', False):
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: rgba(102, 126, 234, 0.1); border-radius: 10px; margin: 10px 0;">
            <h3>🎤 Listening...</h3>
            <p>Speak now. Click "Stop Recording" when finished.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add Web Speech API JavaScript
        speech_recognition_js = """
        <script>
        if ('webkitSpeechRecognition' in window) {
            const recognition = new webkitSpeechRecognition();
            recognition.continuous = false;
            recognition.interimResults = false;
            recognition.lang = 'en-US';
            
            recognition.onresult = function(event) {
                const transcript = event.results[0][0].transcript;
                // Store the transcript in a hidden input
                const hiddenInput = document.createElement('input');
                hiddenInput.type = 'hidden';
                hiddenInput.id = 'speech_result';
                hiddenInput.value = transcript;
                document.body.appendChild(hiddenInput);
                
                // Show the transcript to user
                const resultDiv = document.createElement('div');
                resultDiv.innerHTML = '<h4>Speech Recognized:</h4><p>' + transcript + '</p>';
                resultDiv.style.background = '#e8f5e8';
                resultDiv.style.padding = '10px';
                resultDiv.style.borderRadius = '5px';
                resultDiv.style.margin = '10px 0';
                document.body.appendChild(resultDiv);
            };
            
            recognition.onerror = function(event) {
                console.error('Speech recognition error:', event.error);
                alert('Speech recognition error: ' + event.error);
            };
            
            recognition.start();
        } else {
            alert('Speech recognition not supported in this browser');
        }
        </script>
        """
        st.markdown(speech_recognition_js, unsafe_allow_html=True)
        
        if st.button("⏹️ Stop Recording", key="stop_recording"):
            st.session_state.voice_recording = False
            st.rerun()
        
        # Text input for manual entry (fallback)
        voice_text = st.text_input("Or type your message:", placeholder="Speech will appear here automatically...")
        if voice_text:
            # Process the voice input as regular chat
            return voice_text

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
    
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    
    if 'file_memories' not in st.session_state:
        st.session_state.file_memories = {}
    
    if 'show_live_thoughts' not in st.session_state:
        st.session_state.show_live_thoughts = True
    
    if 'thought_stream_data' not in st.session_state:
        st.session_state.thought_stream_data = []
    
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False
    
    if 'voice_mode_enabled' not in st.session_state:
        st.session_state.voice_mode_enabled = False

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

def render_advanced_consciousness_controls():
    """Render advanced consciousness parameter controls"""
    st.subheader("🎛️ Consciousness Controls")
    
    if st.session_state.ai.consciousness_enabled:
        consciousness = st.session_state.ai.consciousness
        
        with st.expander("Real-time Consciousness Parameters", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🧠 Thinking Parameters**")
                thinking_speed = st.slider("Thinking Speed (s)", 0.1, 2.0, 0.5, 0.1, 
                                         help="How fast Sentient processes thoughts")
                memory_threshold = st.slider("Memory Threshold", 0.1, 0.9, 0.3, 0.1,
                                           help="Minimum importance to form memories")
                temperature = st.slider("Temperature", 0.5, 1.5, 1.0, 0.1,
                                       help="Creativity vs consistency balance")
            
            with col2:
                st.markdown("**⚡ Drive Priorities**")
                curiosity_weight = st.slider("Curiosity Drive", 0.0, 1.0, 0.7, 0.1)
                creativity_weight = st.slider("Creative Drive", 0.0, 1.0, 0.6, 0.1)
                coherence_weight = st.slider("Coherence Drive", 0.0, 1.0, 0.8, 0.1)
                contribution_weight = st.slider("Contribution Drive", 0.0, 1.0, 0.5, 0.1)
            
            if st.button("Apply Real-time Updates"):
                # Update consciousness parameters
                consciousness.update_drives(
                    curiosity=curiosity_weight,
                    growth=creativity_weight,
                    coherence=coherence_weight,
                    contribution=contribution_weight
                )
                
                # Store other parameters in session state for use during generation
                st.session_state.consciousness_params = {
                    'thinking_speed': thinking_speed,
                    'memory_threshold': memory_threshold,
                    'temperature': temperature
                }
                
                st.success("🎯 Consciousness parameters updated in real-time!")
                st.rerun()

def render_chat_interface():
    """Render the main chat interface"""
    st.subheader("💬 Chat with Sentient")
    
    # Voice controls
    render_voice_controls()
    
    # Display options (no more mode selection needed!)
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("**🧠 Sentient Mode: Always Conscious**")
        st.caption("Natural consciousness that adapts to every conversation")
    
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
                result = st.session_state.ai.generate(user_input)
                
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
    
    # Live thought ticker at top
    render_live_thought_ticker()
    
    # Header with theme toggle
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        if st.button("🌓", help="Toggle Dark/Light Mode"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
    
    with col2:
        st.markdown('<h1 class="main-header">🧠 Sentient</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; color: #666; font-size: 1.1rem;">The AI that actually remembers and thinks continuously</p>', unsafe_allow_html=True)
    
    with col3:
        # Quick stats
        if st.session_state.ai.consciousness_enabled:
            consciousness_data = st.session_state.ai.get_live_consciousness_data()
            recent_thoughts = consciousness_data.get('recent_thoughts', [])
            st.metric("", f"{len(recent_thoughts)} thoughts", label_visibility="collapsed")
    
    # Sidebar
    with st.sidebar:
        st.header("🌟 Consciousness Monitor")
        
        # Auto-refresh toggle
        auto_refresh = st.toggle("Auto-refresh", value=False)
        if auto_refresh:
            time.sleep(2)
            st.rerun()
        
        # Live thought stream controls
        st.session_state.show_live_thoughts = st.checkbox("Show Live Thought Ticker", value=True)
        
        # Consciousness panel
        render_consciousness_panel()
        
        # File upload section
        st.divider()
        st.subheader("📁 File Upload")
        render_file_upload_zone()
        render_uploaded_files_sidebar()
        
        st.divider()
        
        # Thought stream
        if st.checkbox("Show Detailed Thoughts", value=True):
            render_thought_stream()
        
        # Memory timeline
        if st.checkbox("Show Memory Timeline"):
            render_memory_timeline()
    
    # Main content tabs for better organization
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "🎛️ Controls", "📊 Analytics", "⚙️ Settings"])
    
    with tab1:
        # Chat interface
        render_chat_interface()
    
    with tab2:
        # Advanced consciousness controls
        render_advanced_consciousness_controls()
        
        st.divider()
        
        # Thought timeline chart
        if st.session_state.ai.consciousness_enabled:
            st.subheader("📈 Thought Timeline")
            thought_chart = create_thought_timeline()
            if thought_chart:
                st.plotly_chart(thought_chart, use_container_width=True)
            else:
                st.info("Start a conversation to see the thought timeline")
    
    with tab3:
        # Live analytics dashboard
        render_live_analytics_dashboard()
    
    with tab4:
        # Session management
        render_session_management()
        
        st.divider()
        
        # Advanced features
        render_advanced_features()

if __name__ == "__main__":
    main()