import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os

# ================= Import Fix =================
try:
    import mediapipe.solutions.pose as mp_pose
    import mediapipe.solutions.drawing_utils as mp_drawing
except ImportError:
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import drawing_utils as mp_drawing

# ================= Page Config =================
st.set_page_config(page_title="Climbing AI Coach", page_icon="🧗", layout="wide")
st.title("🧗 AI Climbing Coach (WebM Stable Edition)")

# ================= Sidebar =================
with st.sidebar:
    st.header("🔧 Settings")
    flag_threshold = st.slider("Flagging Threshold Angle", 130, 170, 150)
    st.divider()
    show_skeleton = st.checkbox("Show Skeleton", value=True)
    show_trail = st.checkbox("Show Hip Trajectory", value=True)
    st.warning("Exporting as WebM for maximum compatibility.")

# ================= Analysis Logic =================
def process_video(input_path, output_path):
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(input_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 30
    
    # --- THE KEY FIX: VP80 Codec ---
    # VP8 is an open-source codec that works on 99% of Linux servers
    fourcc = cv2.VideoWriter_fourcc(*'VP80') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        st.error("Codec Error: Server cannot initialize VideoWriter with VP8.")
        return None

    hip_trail = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # AI Processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            def get_coords(idx):
                return [landmarks[idx].x * width, landmarks[idx].y * height]
            
            # 1. Trajectory
            if show_trail:
                l_hip = get_coords(23)
                current_hip = (int(l_hip[0]), int(l_hip[1]))
                hip_trail.append(current_hip)
                if len(hip_trail) > 100: hip_trail.pop(0)
                for i in range(1, len(hip_trail)):
                    thick = int(np.sqrt(i/len(hip_trail)) * 5) + 1
                    cv2.line(image, hip_trail[i-1], hip_trail[i], (0, 255, 255), thick)
                cv2.circle(image, current_hip, 6, (0, 0, 255), -1)

            # 2. Flagging
            l_hip, l_knee, l_ankle = get_coords(23), get_coords(25), get_coords(27)
            r_hip, r_knee, r_ankle = get_coords(24), get_coords(26), get_coords(28)
            
            def check_leg(h, k, a):
                h_np, k_np, a_np = np.array(h), np.array(k), np.array(a)
                rad = np.arctan2(a_np[1]-k_np[1], a_np[0]-k_np[0]) - np.arctan2(h_np[1]-k_np[1], h_np[0]-k_np[0])
                ang = np.abs(rad*180.0/np.pi)
                if ang > 180.0: ang = 360-ang
                if ang > flag_threshold and abs(a[0]-h[0]) > width * 0.15: return True, int(ang)
                return False, 0

            fl, al = check_leg(l_hip, l_knee, l_ankle)
            fr, ar = check_leg(r_hip, r_knee, r_ankle)
            if fl or fr:
                disp = al if fl else ar
                cv2.putText(image, f"NICE FLAG! ({disp})", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            # 3. Skeleton
            if show_skeleton:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Write frame to output
        out.write(image)
        
        frame_count += 1
        if frame_count % 10 == 0: # Update UI every 10 frames to save speed
            progress_bar.progress(min(frame_count/total_frames, 1.0))
            status_text.text(f"Processing frame {frame_count}/{
