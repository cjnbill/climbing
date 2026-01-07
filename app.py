import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os

# ================= 🛠️ Stable Imports =================
try:
    import mediapipe.solutions.pose as mp_pose
    import mediapipe.solutions.drawing_utils as mp_drawing
except ImportError:
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import drawing_utils as mp_drawing

# ================= Page Config =================
st.set_page_config(page_title="Climbing AI Coach", page_icon="🧗", layout="wide")

st.title("🧗 AI Climbing Coach (Turbo Edition)")
st.markdown("---")

# ================= Sidebar: Optimization & Settings =================
with st.sidebar:
    st.header("🔧 Settings")
    
    # NEW: Frame Skip Control to speed up processing
    processing_speed = st.select_slider(
        "Processing Speed",
        options=["Standard", "Fast", "Turbo"],
        value="Fast"
    )
    speed_map = {"Standard": 0, "Fast": 2, "Turbo": 4}
    frame_skip = speed_map[processing_speed]
    
    st.divider()
    flag_threshold = st.slider("Flagging Threshold Angle", 130, 170, 150)
    show_skeleton = st.checkbox("Show Skeleton", value=True)
    show_trail = st.checkbox("Show Hip Trajectory", value=True)
    
    st.info(f"Speed set to {processing_speed}. Skipping {frame_skip} frames between AI calculations.")

# ================= Core Analysis Logic =================
def process_video(input_path, output_path, skip_count):
    # model_complexity=0 is the fastest model for weak CPUs
    pose = mp_pose.Pose(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 30
    
    # VP80 Codec for maximum Cloud Linux compatibility
    fourcc = cv2.VideoWriter_fourcc(*'VP80') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    hip_trail = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # SPEED OPTIMIZATION: Skip AI processing for intermediate frames
        if frame_count % (skip_count + 1) != 0:
            out.write(frame)
            frame_count += 1
            continue
        
        # AI Processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            def get_coords(idx):
                return [landmarks[idx].x * width, landmarks[idx].y * height]
            
            # 1. Hip Trajectory
            if show_trail:
                l_hip = get_coords(23)
                current
