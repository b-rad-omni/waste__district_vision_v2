import streamlit as st
import os
import json
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

# Configure page
st.set_page_config(
    page_title="Security Detection Highlights",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1f2937;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 500;
        color: #374151;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .video-container {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: #f9fafb;
    }
    .detection-badge {
        background: #10b981;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.25rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

def load_detection_data():
    """Load detection data from clips folder"""
    clips_folder = Path("clips")  # Point to your clips folder
    
    # Get all image/video files
    image_files = []
    for ext in ['*.jpg', '*.png', '*.mp4', '*.avi']:
        image_files.extend(clips_folder.glob(ext))
    
    # Organize by time (you can customize this logic)
    daily_clips = []
    weekly_clips = []
    monthly_clips = []
    
    for file_path in image_files:
        # Extract timestamp from filename or file creation time
        file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
        
        clip_data = {
            'file': str(file_path),
            'score': 0.95,  # You could extract from filename
            'objects': ['large foreign object'],
            'time': file_time.strftime('%H:%M:%S'),
            'timestamp': file_time
        }
        
        # Categorize by age
        now = datetime.now()
        if (now - file_time).days == 0:
            daily_clips.append(clip_data)
        elif (now - file_time).days <= 7:
            weekly_clips.append(clip_data)
        elif (now - file_time).days <= 30:
            monthly_clips.append(clip_data)
    
    return {
        'daily': daily_clips,
        'weekly': weekly_clips,
        'monthly': monthly_clips
    }

def create_video_player(file_path, metadata):
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if file_path.endswith(('.mp4', '.avi', '.mkv')):
            st.video(file_path)
        elif file_path.endswith(('.jpg', '.png')):
            st.image(file_path, use_container_width=True)
        else:
            st.info(f"File: {os.path.basename(file_path)}")
    
    

# Header
st.markdown('<h1 class="main-header">🎯 Security Detection Highlights</h1>', unsafe_allow_html=True)

# Load data
data = load_detection_data()

# Summary metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-card"><h3>Today</h3><h2>{}</h2><p>Detections</p></div>'.format(len(data['daily'])), unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card"><h3>This Week</h3><h2>{}</h2><p>Top Clips</p></div>'.format(len(data['weekly'])), unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card"><h3>This Month</h3><h2>{}</h2><p>Highlights</p></div>'.format(len(data['monthly'])), unsafe_allow_html=True)
with col4:
    avg_score = sum([item['score'] for sublist in data.values() for item in sublist]) / sum(len(sublist) for sublist in data.values())
    st.markdown('<div class="metric-card"><h3>Avg Score</h3><h2>{:.0%}</h2><p>Confidence</p></div>'.format(avg_score), unsafe_allow_html=True)

# Navigation tabs
tab1, tab2, tab3 = st.tabs(["📅 Today's Highlights", "📊 Weekly Top Clips", "🏆 Monthly Best"])

with tab1:
    st.markdown('<h2 class="section-header">Today\'s Detection Highlights</h2>', unsafe_allow_html=True)
    if data['daily']:
        for detection in data['daily']:
            with st.container():
                st.markdown('<div class="video-container">', unsafe_allow_html=True)
                create_video_player(detection['file'], detection)
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No detections recorded for today.")

with tab2:
    st.markdown('<h2 class="section-header">This Week\'s Most Interesting Clips</h2>', unsafe_allow_html=True)
    if data['weekly']:
        for detection in data['weekly']:
            with st.container():
                st.markdown('<div class="video-container">', unsafe_allow_html=True)
                create_video_player(detection['file'], detection)
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No weekly highlights available.")

with tab3:
    st.markdown('<h2 class="section-header">Monthly Highlights</h2>', unsafe_allow_html=True)
    if data['monthly']:
        for detection in data['monthly']:
            with st.container():
                st.markdown('<div class="video-container">', unsafe_allow_html=True)
                create_video_player(detection['file'], detection)
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No monthly highlights available.")

# Footer
st.markdown("---")
st.markdown("*Detection system powered by YOLO8 + ByteTrack*")