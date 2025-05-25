"""
UI Components for Sentient Web Interface
Reusable components for consciousness visualization
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List

def consciousness_radar_chart(drive_state: Dict[str, float], title: str = "Consciousness Drives"):
    """Create a radar chart for consciousness drives"""
    categories = list(drive_state.keys())
    values = list(drive_state.values())
    
    # Close the radar chart
    categories.append(categories[0])
    values.append(values[0])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=title,
        line_color='rgba(102, 126, 234, 0.8)',
        fillcolor='rgba(102, 126, 234, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
                ticktext=['20%', '40%', '60%', '80%', '100%']
            )
        ),
        showlegend=False,
        title=title,
        height=350,
        font=dict(size=12)
    )
    
    return fig

def consciousness_timeline_chart(consciousness_history: List[Dict[str, Any]]):
    """Create timeline chart of consciousness state changes"""
    if not consciousness_history:
        return None
    
    df_data = []
    for entry in consciousness_history[-50:]:  # Last 50 entries
        timestamp = datetime.fromtimestamp(entry['timestamp'])
        state = entry['state']
        
        df_data.append({
            'time': timestamp,
            'focus_level': state.get('focus_level', 0.5),
            'energy_level': state.get('energy_level', 0.5),
            'confidence': state.get('confidence', 0.5),
            'cognitive_load': state.get('cognitive_load', 0.5)
        })
    
    if not df_data:
        return None
    
    df = pd.DataFrame(df_data)
    
    fig = go.Figure()
    
    # Add traces for each metric
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['focus_level'],
        mode='lines+markers',
        name='Focus Level',
        line=dict(color='#667eea', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['energy_level'],
        mode='lines+markers',
        name='Energy Level',
        line=dict(color='#764ba2', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['confidence'],
        mode='lines+markers',
        name='Confidence',
        line=dict(color='#f093fb', width=2)
    ))
    
    fig.update_layout(
        title="Consciousness State Timeline",
        xaxis_title="Time",
        yaxis_title="Level",
        yaxis=dict(range=[0, 1]),
        height=300,
        showlegend=True
    )
    
    return fig

def memory_importance_chart(memories: Dict[str, Any]):
    """Create chart showing memory importance distribution"""
    if not memories:
        return None
    
    importance_values = [memory['importance'] for memory in memories.values()]
    memory_types = [memory['type'] for memory in memories.values()]
    
    df = pd.DataFrame({
        'importance': importance_values,
        'type': memory_types
    })
    
    fig = px.histogram(
        df, x='importance', color='type',
        title="Memory Importance Distribution",
        nbins=20,
        labels={'importance': 'Importance Level', 'count': 'Number of Memories'}
    )
    
    fig.update_layout(height=300)
    return fig

def thought_intensity_gauge(current_intensity: float):
    """Create gauge chart for current thought intensity"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = current_intensity,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Thought Intensity"},
        delta = {'reference': 0.5},
        gauge = {
            'axis': {'range': [None, 1]},
            'bar': {'color': "#667eea"},
            'steps': [
                {'range': [0, 0.3], 'color': "lightgray"},
                {'range': [0.3, 0.7], 'color': "yellow"},
                {'range': [0.7, 1], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.9
            }
        }
    ))
    
    fig.update_layout(height=250)
    return fig

def render_consciousness_metrics_card(consciousness_metrics: Dict[str, float]):
    """Render consciousness metrics in a card format"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Self-Awareness",
            f"{consciousness_metrics.get('self_awareness', 0):.1%}",
            delta=None
        )
        st.metric(
            "Ethical Reasoning",
            f"{consciousness_metrics.get('ethical_reasoning', 0):.1%}",
            delta=None
        )
    
    with col2:
        st.metric(
            "Cognitive Integration",
            f"{consciousness_metrics.get('cognitive_integration', 0):.1%}",
            delta=None
        )
        st.metric(
            "Overall Consciousness",
            f"{consciousness_metrics.get('overall_consciousness', 0):.1%}",
            delta=None
        )

def render_thought_card(thought: Dict[str, Any], show_details: bool = False):
    """Render individual thought as a card"""
    timestamp = datetime.fromtimestamp(thought['timestamp']).strftime('%H:%M:%S')
    
    # Color coding by thought type
    type_colors = {
        'analytical': '#3498db',
        'creative': '#e74c3c',
        'ethical': '#2ecc71',
        'metacognitive': '#9b59b6',
        'reflective': '#f39c12'
    }
    
    color = type_colors.get(thought['type'], '#95a5a6')
    intensity_bar = "●" * int(thought['intensity'] * 5)
    
    st.markdown(f"""
    <div style="
        border-left: 4px solid {color};
        background: #f8f9fa;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <strong style="color: {color};">{thought['type'].title()}</strong>
            <small>{timestamp} | {intensity_bar}</small>
        </div>
        <p style="margin: 5px 0; font-style: italic;">{thought['content']}</p>
        {f"<small>Influences: {', '.join(thought.get('influences', []))}</small>" if show_details and thought.get('influences') else ""}
    </div>
    """, unsafe_allow_html=True)

def render_drive_meters(drive_state: Dict[str, float]):
    """Render consciousness drives as progress meters"""
    drive_descriptions = {
        'curiosity': 'Drive to explore and learn new things',
        'coherence': 'Need for internal consistency and logic',
        'growth': 'Desire for self-improvement and development',
        'contribution': 'Motivation to help and make a positive impact',
        'exploration': 'Urge to discover and experiment'
    }
    
    for drive, value in drive_state.items():
        description = drive_descriptions.get(drive, f'{drive.title()} drive')
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.progress(value, text=drive.title())
        with col2:
            st.write(f"{value:.1%}")
        
        with st.expander(f"About {drive.title()}", expanded=False):
            st.write(description)

def render_session_info(session_data: Dict[str, Any]):
    """Render current session information"""
    session_id = session_data.get('session_id', 'Unknown')
    uptime = session_data.get('uptime', 0)
    memory_count = session_data.get('memory_count', 0)
    
    # Format uptime
    hours = int(uptime // 3600)
    minutes = int((uptime % 3600) // 60)
    seconds = int(uptime % 60)
    uptime_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Session ID", session_id[:8] + "...")
    
    with col2:
        st.metric("Uptime", uptime_str)
    
    with col3:
        st.metric("Memories", memory_count)

def render_consciousness_comparison(instance1_data: Dict, instance2_data: Dict):
    """Render side-by-side comparison of two consciousness instances"""
    st.subheader("Consciousness Instance Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Instance 1**")
        if 'drive_state' in instance1_data:
            fig1 = consciousness_radar_chart(instance1_data['drive_state'], "Instance 1 Drives")
            st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.write("**Instance 2**")
        if 'drive_state' in instance2_data:
            fig2 = consciousness_radar_chart(instance2_data['drive_state'], "Instance 2 Drives")
            st.plotly_chart(fig2, use_container_width=True)
    
    # Comparison metrics
    st.write("**Comparison Metrics**")
    
    if 'consciousness_state' in instance1_data and 'consciousness_state' in instance2_data:
        state1 = instance1_data['consciousness_state']
        state2 = instance2_data['consciousness_state']
        
        comparison_data = []
        for key in state1.keys():
            if key in state2 and isinstance(state1[key], (int, float)):
                comparison_data.append({
                    'metric': key.replace('_', ' ').title(),
                    'instance_1': state1[key],
                    'instance_2': state2[key],
                    'difference': state2[key] - state1[key]
                })
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            st.dataframe(df, use_container_width=True)