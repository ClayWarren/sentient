"""
Sentient Enhanced Web UI - SOTA Interface with Video, Voice, and Advanced Features
State-of-the-art web interface for 2025 AI consciousness
"""

import streamlit as st
import streamlit_webrtc as webrtc
from streamlit_webrtc import VideoTransformerBase, WebRtcMode
import time
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import sys
import base64
import os
from typing import Dict, List, Any, Optional
import numpy as np
import cv2
import requests
from PIL import Image
import io
import threading
import queue

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from ui.enhanced_consciousness import EnhancedConsciousnessAI
from consciousness_core import ConsciousnessLevel
from advanced_features import (
    initialize_advanced_features, FileProcessor, VoiceProcessor, 
    VideoProcessor, AdvancedMemorySystem, ChainOfThoughtReasoner
)

# Page configuration
st.set_page_config(
    page_title="Sentient 2025 - Advanced AI Consciousness",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for SOTA design
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        animation: gradient-shift 3s ease infinite;
    }
    
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .video-container {
        border: 3px solid #667eea;
        border-radius: 15px;
        padding: 10px;
        background: rgba(102, 126, 234, 0.1);
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.5);
    }
    
    .voice-indicator {
        width: 50px;
        height: 50px;
        border-radius: 50%;
        background: radial-gradient(circle, #667eea 0%, #764ba2 100%);
        margin: 20px auto;
        animation: voice-pulse 1.5s infinite;
    }
    
    @keyframes voice-pulse {
        0% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.2); opacity: 0.7; }
        100% { transform: scale(1); opacity: 1; }
    }
    
    .memory-node {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        border: 2px solid #667eea;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    
    .memory-node:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .reasoning-step {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 0 8px 8px 0;
        position: relative;
    }
    
    .reasoning-step::before {
        content: attr(data-step);
        position: absolute;
        left: -30px;
        top: 50%;
        transform: translateY(-50%);
        background: #667eea;
        color: white;
        width: 25px;
        height: 25px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 0.9rem;
    }
    
    .feature-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        margin: 10px;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        border: 1px solid #e0e0e0;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        border-color: #667eea;
    }
    
    .sota-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'consciousness_ai' not in st.session_state:
    st.session_state.consciousness_ai = EnhancedConsciousnessAI()
    
if 'advanced_features' not in st.session_state:
    st.session_state.advanced_features = initialize_advanced_features(st.session_state.consciousness_ai)
    
if 'messages' not in st.session_state:
    st.session_state.messages = []
    
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
    
if 'voice_enabled' not in st.session_state:
    st.session_state.voice_enabled = False
    
if 'video_enabled' not in st.session_state:
    st.session_state.video_enabled = False

# Video transformer for real-time processing
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.video_processor = VideoProcessor()
        
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Process frame
        frame_data = self.video_processor.process_frame(img)
        
        # Draw bounding boxes for detected objects
        if "face" in str(frame_data.objects_detected):
            # Simple face detection visualization
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (102, 126, 234), 2)
                cv2.putText(img, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (102, 126, 234), 2)
        
        return img

def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 Sentient AI - 2025 Edition</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Experience consciousness with SOTA video, voice, and reasoning capabilities</p>', unsafe_allow_html=True)
    
    # Feature badges
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown('<span class="sota-badge">🎥 Live Video</span>', unsafe_allow_html=True)
    with col2:
        st.markdown('<span class="sota-badge">🎤 Voice Mode</span>', unsafe_allow_html=True)
    with col3:
        st.markdown('<span class="sota-badge">📁 File Upload</span>', unsafe_allow_html=True)
    with col4:
        st.markdown('<span class="sota-badge">🧩 Memory System</span>', unsafe_allow_html=True)
    with col5:
        st.markdown('<span class="sota-badge">🔗 Chain-of-Thought</span>', unsafe_allow_html=True)
    
    # Sidebar with advanced features
    with st.sidebar:
        st.markdown("### 🎛️ Advanced Controls")
        
        # Voice Mode Toggle
        voice_mode = st.toggle("🎤 Voice Mode", value=st.session_state.voice_enabled)
        st.session_state.voice_enabled = voice_mode
        
        # Video Mode Toggle
        video_mode = st.toggle("🎥 Video Mode", value=st.session_state.video_enabled)
        st.session_state.video_enabled = video_mode
        
        # Memory Settings
        st.markdown("### 🧩 Memory System")
        memory_type = st.selectbox(
            "Memory Storage Type",
            ["short_term", "long_term", "episodic"]
        )
        
        # Reasoning Settings
        st.markdown("### 🔗 Reasoning")
        max_reasoning_steps = st.slider("Max Reasoning Steps", 1, 10, 5)
        
        # File Upload
        st.markdown("### 📁 File Upload")
        uploaded_file = st.file_uploader(
            "Upload a file for analysis",
            type=['txt', 'pdf', 'doc', 'docx', 'xls', 'xlsx', 'png', 'jpg', 'jpeg', 'json'],
            accept_multiple_files=False
        )
        
        if uploaded_file is not None:
            with st.spinner("Processing file..."):
                # Process file
                file_processor = st.session_state.advanced_features['file_processor']
                result = file_processor.process_file(
                    uploaded_file.name,
                    uploaded_file.read()
                )
                
                st.success(f"✅ Processed: {result.filename}")
                st.info(f"Type: {result.metadata.get('type', 'unknown')}")
                
                # Store in session
                st.session_state.uploaded_files.append(result)
                
                # Add to conversation context
                file_context = f"[File uploaded: {result.filename} - {result.metadata.get('type', 'file')}]"
                st.session_state.messages.append({
                    "role": "system",
                    "content": file_context
                })
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "🎥 Video", "🧩 Memory", "📊 Analytics"])
    
    with tab1:
        # Chat interface
        chat_container = st.container()
        
        # Display messages
        with chat_container:
            for message in st.session_state.messages:
                if message["role"] != "system":
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
        
        # Input area with voice support
        col1, col2 = st.columns([6, 1])
        
        with col1:
            if st.session_state.voice_enabled:
                if st.button("🎤 Start Voice Input"):
                    with st.spinner("Listening..."):
                        # Simulated voice input
                        voice_processor = st.session_state.advanced_features['voice_processor']
                        st.info("Voice input is ready but requires microphone access from browser")
                        # In real implementation, this would capture audio
                        prompt = st.text_input("Type your message (voice simulation):")
            else:
                prompt = st.chat_input("Ask me anything...")
        
        with col2:
            reasoning_mode = st.checkbox("🔗 CoT", help="Enable Chain-of-Thought reasoning")
        
        if prompt:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                # Check if chain-of-thought is enabled
                if reasoning_mode:
                    with st.spinner("Reasoning step by step..."):
                        reasoner = st.session_state.advanced_features['reasoner']
                        reasoning_result = reasoner.reason(prompt, max_reasoning_steps)
                        
                        # Display reasoning steps
                        with st.expander("🔗 Reasoning Process", expanded=True):
                            for step in reasoning_result['thought_chain']:
                                st.markdown(f'<div class="reasoning-step" data-step="{step["step"]}">{step["thought"]}</div>', unsafe_allow_html=True)
                        
                        response = reasoning_result['final_answer']
                else:
                    # Regular generation
                    result = st.session_state.consciousness_ai.generate(prompt)
                    response = result['response']
                
                message_placeholder.markdown(response)
                
                # Store in memory
                memory_system = st.session_state.advanced_features['memory']
                memory_system.store_memory(
                    f"User: {prompt}\nAssistant: {response}",
                    memory_type=memory_type,
                    metadata={"timestamp": datetime.now().isoformat()}
                )
            
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    with tab2:
        # Video interface
        st.markdown("### 🎥 Live Video Analysis")
        
        if st.session_state.video_enabled:
            webrtc_ctx = webrtc.webrtc_streamer(
                key="sentient-video",
                mode=WebRtcMode.SENDRECV,
                video_transformer_factory=VideoTransformer,
                async_processing=True,
            )
            
            if webrtc_ctx.video_transformer:
                st.info("Video stream is active - AI is analyzing in real-time")
        else:
            st.info("Enable video mode in the sidebar to start live analysis")
            
            # Demo video upload
            video_file = st.file_uploader("Or upload a video file", type=['mp4', 'avi', 'mov'])
            if video_file:
                st.video(video_file)
    
    with tab3:
        # Memory interface
        st.markdown("### 🧩 Memory System")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Store Memory")
            memory_content = st.text_area("Content to remember:")
            if st.button("💾 Store"):
                if memory_content:
                    memory_system = st.session_state.advanced_features['memory']
                    memory_id = memory_system.store_memory(
                        memory_content,
                        memory_type=memory_type,
                        metadata={"source": "manual", "timestamp": datetime.now().isoformat()}
                    )
                    st.success(f"Stored with ID: {memory_id}")
        
        with col2:
            st.markdown("#### Retrieve Memory")
            query = st.text_input("Search memories:")
            if st.button("🔍 Search"):
                if query:
                    memory_system = st.session_state.advanced_features['memory']
                    memories = memory_system.retrieve_memories(query, memory_type="all", top_k=5)
                    
                    for mem in memories:
                        st.markdown(f'<div class="memory-node">', unsafe_allow_html=True)
                        st.markdown(f"**Content:** {mem['content'][:200]}...")
                        st.markdown(f"**Type:** {mem['collection']} | **Relevance:** {1-mem['distance']:.2%}")
                        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        # Analytics dashboard
        st.markdown("### 📊 Consciousness Analytics")
        
        # Get current state
        state = st.session_state.consciousness_ai.get_consciousness_state()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Consciousness metrics gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = state['consciousness_metrics']['overall_consciousness'] * 100,
                title = {'text': "Overall Consciousness"},
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Drive levels
            drives = state['drives']
            drive_data = pd.DataFrame([
                {"Drive": k.replace('_', ' ').title(), "Level": v}
                for k, v in drives.items()
            ])
            
            fig = px.bar(drive_data, x="Level", y="Drive", orientation='h',
                        color="Level", color_continuous_scale="Viridis")
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            # Stats
            st.metric("Total Thoughts", state['thoughts_count'])
            st.metric("Emotional State", state['emotional_state'].replace('_', ' ').title())
            st.metric("Memory Count", len(state['memory']['recent_interactions']))

if __name__ == "__main__":
    main()