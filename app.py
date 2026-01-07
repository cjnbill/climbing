import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os
import shutil

# ================= 🛠️ Critical Permission Fix =================
# We force MediaPipe to use a writable directory for its models
import mediapipe.python.solutions.pose as mp_pose_mod
import mediapipe.python.solutions.download_utils as mp_download_mod

# 1. Define a writable path in the temporary directory
writable_dir = os.path.join(tempfile.gettempdir(), "mediapipe_models")
os.makedirs(writable_dir, exist_ok=True)

# 2. Monkey-patch the internal path so MediaPipe thinks its home is in /tmp
# This redirects the download from the read-only site-packages to a writable folder
mp_download_mod._get_model_abspath = lambda path: os.path.join(writable_dir, os.path.basename(path))

# Standard Imports
try:
    import mediapipe.solutions.pose as mp_pose
    import mediapipe.solutions.drawing_utils as mp_drawing
except ImportError:
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import drawing_utils as mp_drawing
# =============================================================

st.set_page_config(page_title="Climbing AI Coach", page_icon="🧗", layout="wide")
st.title("🧗 AI Climbing Coach (Permission Fixed)")

with st.sidebar:
    st.header("🔧 Settings")
    processing_speed = st.select_slider("Speed", options=["Standard", "Fast", "Turbo"], value="Fast")
    speed_map = {"Standard": 0, "Fast": 2, "Turbo": 4}
    frame_skip = speed_map[processing_speed]
    st.divider()
    flag_threshold = st.slider("Flagging Threshold", 130, 170, 150)
    show_skeleton = st.checkbox("Show Skeleton", value=True)
    show_trail = st.checkbox("Show Hip Trajectory", value=True)
    show_coaching = st.checkbox("Coaching Alerts", value=True)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    ang = np.abs(rad * 180.0 / np.pi)
    return 360 - ang if ang > 180 else ang

def process_video(input_path, output_path, skip_count):
    # Initialize Pose - with redirected path, this will now download to /tmp successfully
    pose = mp_pose.Pose(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    
    fourcc = cv2.VideoWriter_fourcc(*'VP80') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    hip_trail = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        if frame_count % (skip_count + 1) != 0:
            out.write(frame)
            frame_count += 1
            continue

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            def get_pt(idx): return [lm[idx].x * width, lm[idx].y * height]
            
            if show_trail:
                hip = get_pt(23)
                hip_trail.append((int(hip[0]), int(hip[1])))
                if len(hip_trail) > 60: hip_trail.pop(0)
                for i in range(1, len(hip_trail)):
                    cv2.line(image, hip_trail[i-1], hip_trail[i], (0, 255, 255), 2)

            l_hip, l_knee, l_ankle = get_pt(23), get_pt(25), get_pt(27)
            r_hip, r_knee, r_ankle = get_pt(24), get_pt(26), get_pt(28)
            if calculate_angle(l_hip, l_knee, l_ankle) > flag_threshold or calculate_angle(r_hip, r_knee, r_ankle) > flag_threshold:
                cv2.putText(image, "NICE FLAG!", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            if show_coaching:
                l_sh, l_el, l_wr = get_pt(11), get_pt(13), get_pt(15)
                r_sh, r_el, r_wr = get_pt(12), get_pt(14), get_pt(16)
                if calculate_angle(l_sh, l_el, l_wr) < 90 or calculate_angle(r_sh, r_el, r_wr) < 90:
                    cv2.putText(image, "KEEP ARMS STRAIGHT!", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            if show_skeleton:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        out.write(image)
        frame_count += 1
        if frame_count % 15 == 0: progress_bar.progress(min(frame_count/total_frames, 1.0))

    cap.release()
    out.release()
    progress_bar.empty()
    return output_path

# ================= UI Layout =================
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("1. Video Source")
    if 'processed_video' in st.session_state:
        if st.button("🔄 Analyze New"):
            for k in ['processed_video', 'original_video']: st.session_state.pop(k, None)
            st.rerun()
    uploaded_file = st.file_uploader("Upload MOV/MP4", type=['mov', 'mp4'])

if uploaded_file:
    if 'original_video' not in st.session_state:
        t = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
        t.write(uploaded_file.read())
        st.session_state['original_video'] = t.name
    
    with col1:
        st.video(st.session_state['original_video'])
        if 'processed_video' not in st.session_state:
            if st.button("Start AI Analysis 🚀", type="primary"):
                out = tempfile.NamedTemporaryFile(delete=False, suffix='.webm').name
                with col2:
                    with st.spinner('Analyzing...'):
                        res = process_video(st.session_state['original_video'], out, frame_skip)
                    if res and os.path.getsize(res) > 0:
                        st.session_state['processed_video'] = res
                        st.rerun()

if 'processed_video' in st.session_state:
    with col2:
        st.subheader("2. AI Result")
        res = st.session_state['processed_video']
        with open(res, 'rb') as f: st.video(f.read(), format="video/webm")
        st.download_button("📥 Download", open(res, 'rb'), "climb.webm")
