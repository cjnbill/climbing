import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os

# ================= 标准导入 =================
# 只要 requirements.txt 设置正确，这里就不会报错
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

st.set_page_config(page_title="Climbing AI Coach", page_icon="🧗", layout="wide")
st.title("🧗 AI Climbing Coach")
st.markdown("---")

with st.sidebar:
    st.header("🔧 Settings")
    flag_threshold = st.slider("Flagging Threshold", 130, 170, 150)
    st.divider()
    show_skeleton = st.checkbox("Show Skeleton", value=True)
    show_trail = st.checkbox("Show Hip Trajectory", value=True)

def process_video(input_path, output_path):
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(input_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 30
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    hip_trail = []
    progress_bar = st.progress(0, text="Analyzing...")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            def get_coords(idx):
                return [landmarks[idx].x * width, landmarks[idx].y * height]
            
            # Trajectory
            if show_trail:
                l_hip = get_coords(23)
                current_hip = (int(l_hip[0]), int(l_hip[1]))
                hip_trail.append(current_hip)
                if len(hip_trail) > 100: hip_trail.pop(0)
                for i in range(1, len(hip_trail)):
                    thick = int(np.sqrt(i/len(hip_trail)) * 5) + 1
                    cv2.line(image, hip_trail[i-1], hip_trail[i], (0, 255, 255), thick)
                cv2.circle(image, current_hip, 6, (0, 0, 255), -1)

            # Flagging
            l_hip, l_knee, l_ankle = get_coords(23), get_coords(25), get_coords(27)
            r_hip, r_knee, r_ankle = get_coords(24), get_coords(26), get_coords(28)
            
            def check_leg(h, k, a):
                h, k, a_np = np.array(h), np.array(k), np.array(a)
                rad = np.arctan2(a_np[1]-k[1], a_np[0]-k[0]) - np.arctan2(h[1]-k[1], h[0]-k[0])
                ang = np.abs(rad*180.0/np.pi)
                if ang > 180.0: ang = 360-ang
                if ang > flag_threshold and abs(a[0]-h[0]) > width * 0.15: return True, int(ang)
                return False, 0

            fl, al = check_leg(l_hip, l_knee, l_ankle)
            fr, ar = check_leg(r_hip, r_knee, r_ankle)
            if fl or fr:
                disp = al if fl else ar
                cv2.putText(image, f"NICE FLAG! ({disp})", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            if show_skeleton:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        out.write(image)
        frame_count += 1
        if total_frames > 0: progress_bar.progress(min(frame_count/total_frames, 1.0))

    cap.release()
    out.release()
    progress_bar.empty()
    return output_path

col1, col2 = st.columns([1, 1])
with col1:
    uploaded_file = st.file_uploader("Upload Video", type=['mov', 'mp4'])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    with col1: st.video(tfile.name)
    if st.button("Start Analysis 🚀", type="primary"):
        out_path = tfile.name.replace(".mp4", "_out.mp4")
        with col2:
            with st.spinner('AI Processing...'):
                res = process_video(tfile.name, out_path)
            st.success("Analysis Complete!")
            st.video(res)
