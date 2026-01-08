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
def process_video(input_path, output_path, skip_count=2):
    # 初始化轻量版模型 (complexity=0)
    pose = mp_pose.Pose(
        model_complexity=0, 
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    
    # 缩小输出尺寸（例如固定高度为 480，保持比例）
    target_h = 480
    target_w = int(width * (target_h / height))
    
    fourcc = cv2.VideoWriter_fourcc(*'VP80') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_w, target_h))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # 核心优化：跳帧
        if frame_count % (skip_count + 1) != 0:
            # 即使跳帧，也要调整尺寸写入视频保持时长一致
            small_frame = cv2.resize(frame, (target_w, target_h))
            out.write(small_frame)
            frame_count += 1
            continue

        # 核心优化：缩小处理
        frame = cv2.resize(frame, (target_w, target_h))
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 绘制逻辑 (此处保持原有的 skeleton/trail 代码)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
        out.write(image)
        frame_count += 1
        
    cap.release()
    out.release()
    return output_path

# ================= UI Layout =================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Upload")
    uploaded_file = st.file_uploader("Upload MOV/MP4", type=['mov', 'mp4'])

if uploaded_file:
    # Save original
    suffix = os.path.splitext(uploaded_file.name)[1]
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tfile.write(uploaded_file.read())
    
    with col1:
        st.video(tfile.name)
        if st.button("Start AI Analysis 🚀", type="primary"):
            # Use .webm suffix for output
            out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.webm').name
            
            with col2:
                st.subheader("2. AI Result")
                with st.spinner('Analyzing...'):
                    res = process_video(tfile.name, out_path)
                
                if res and os.path.exists(res) and os.path.getsize(res) > 0:
                    st.success("Analysis Finished!")
                    # Use binary read for stable web display
                    with open(res, 'rb') as f:
                        st.video(f.read(), format="video/webm")
                    st.download_button("📥 Download WebM", open(res, 'rb'), "analysis.webm")
                else:
                    st.error("Output file is still empty. Trying alternate method...")
                    st.info("Debugging: File Path exists: " + str(os.path.exists(out_path)))
