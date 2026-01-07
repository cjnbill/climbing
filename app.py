import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os

# ================= 🛠️ Universal Import Patch =================
# This patch ensures the app works on any Python/MediaPipe version
# by trying multiple import paths for the 'pose' module.
try:
    # Try new structure
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import drawing_utils as mp_drawing
except ImportError:
    try:
        # Try legacy structure
        import mediapipe.solutions.pose as mp_pose
        import mediapipe.solutions.drawing_utils as mp_drawing
    except ImportError:
        # Fallback to top-level import
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
# =============================================================

# ================= Page Config =================
st.set_page_config(page_title="Climbing AI Coach", page_icon="🧗", layout="wide")

st.title("🧗 AI Climbing Coach (Cloud Edition)")
st.markdown("---")

# ================= Sidebar: Settings =================
with st.sidebar:
    st.header("🔧 Settings")
    flag_threshold = st.slider("Flagging Threshold Angle", 130, 170, 150)
    st.caption("Higher angle means straighter leg required.")
    
    st.divider()
    show_skeleton = st.checkbox("Show Skeleton", value=True)
    show_trail = st.checkbox("Show Hip Trajectory", value=True)

# ================= Core Analysis Logic =================
def process_video(input_path, output_path):
    # Use the patched mp_pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 30
    
    # Use 'mp4v' for best compatibility on Linux/Cloud
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    hip_trail = []
    
    # Progress Bar
    progress_bar = st.progress(0, text="AI is analyzing...")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # BGR -> RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            def get_coords(idx):
                return [landmarks[idx].x * width, landmarks[idx].y * height]
            
            # 1. Hip Trajectory (Center of Mass)
            if show_trail:
                l_hip = get_coords(23)
                current_hip = (int(l_hip[0]), int(l_hip[1]))
                hip_trail.append(current_hip)
                if len(hip_trail) > 100: hip_trail.pop(0) # Keep last 100 frames
                
                for i in range(1, len(hip_trail)):
                    # Dynamic line thickness
                    thick = int(np.sqrt(i/len(hip_trail)) * 5) + 1
                    cv2.line(image, hip_trail[i-1], hip_trail[i], (0, 255, 255), thick)
                cv2.circle(image, current_hip, 6, (0, 0, 255), -1)

            # 2. Flagging Detection
            l_hip, l_knee, l_ankle = get_coords(23), get_coords(25), get_coords(27)
            r_hip, r_knee, r_ankle = get_coords(24), get_coords(
