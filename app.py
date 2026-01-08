import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os
import time

# ================= 🛠️ 稳定导入 =================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ================= 页面配置 =================
st.set_page_config(page_title="Climbing AI Coach", page_icon="🧗", layout="wide")
st.title("🧗 AI Climbing Coach (Action Tagging)")

# ================= 侧边栏设置 =================
with st.sidebar:
    st.header("🔧 Settings")
    processing_speed = st.select_slider(
        "Processing Speed", options=["Standard", "Fast", "Turbo"], value="Fast"
    )
    speed_map = {"Standard": 0, "Fast": 2, "Turbo": 4}
    frame_skip = speed_map[processing_speed]
    
    st.divider()
    st.header("📊 Analysis Features")
    show_skeleton = st.checkbox("Show Skeleton", value=True)
    show_metrics = st.checkbox("Tag Key Moves (Speed/Stops)", value=True)
    flag_threshold = st.slider("Flagging Threshold", 130, 170, 150)

# ================= 核心算法 =================
def process_video(input_path, output_path, skip_count):
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(input_path)
    
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    
    # 统一缩放到 720p 以保证处理速度
    target_h = 720
    scale = target_h / orig_h
    width, height = int(orig_w * scale), target_h

    fourcc = cv2.VideoWriter_fourcc(*'VP80') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 用于记录关键动作的变量
    hip_history = []  # 记录最近几帧的髋部位置计算速度
    stops = []        # 记录停顿点 [(x, y, duration), ...]
    last_pos = None
    stop_start_time = None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.resize(frame, (width, height))
        
        # 跳帧逻辑
        if frame_count % (skip_count + 1) != 0:
            out.write(frame)
            frame_count += 1
            continue

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            def get_pt(idx): return np.array([lm[idx].x * width, lm[idx].y * height])
            
            # 1. 获取重心（左右髋部中点）
            l_hip, r_hip = get_pt(23), get_pt(24)
            curr_hip = (l_hip + r_hip) / 2
            
            if show_metrics:
                # --- 关键动作识别：速度与爆发 ---
                if last_pos is not None:
                    dist = np.linalg.norm(curr_hip - last_pos)
                    # 如果向上位移瞬间超过阈值，判定为发力动作
                    if (last_pos[1] - curr_hip[1]) > (height * 0.02): 
                        cv2.putText(image, "POWER MOVE!", (int(curr_hip[0])+20, int(curr_hip[1])), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)
                    
                    # --- 停顿点检测 ---
                    if dist < (width * 0.005): # 几乎没动
                        if stop_start_time is None: stop_start_time = frame_count
                        duration = (frame_count - stop_start_time) / fps
                        if duration > 1.0: # 停顿超过1秒
                            cv2.circle(image, (int(curr_hip[0]), int(curr_hip[1])), 30, (255, 0, 0), 2)
                            cv2.putText(image, f"REST: {duration:.1f}s", (int(curr_hip[0])-40, int(curr_hip[1])-40), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    else:
                        stop_start_time = None
                
                last_pos = curr_hip

            # 2. 绘制骨架
            if show_skeleton:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # 3. 辅助 Flagging 判定
            l_h, l_k, l_a = get_pt(23), get_pt(25), get_pt(27)
            r_h, r_k, r_a = get_pt(24), get_pt(26), get_pt(28)
            def check_ang(a, b, c):
                rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
                ang = np.abs(rad*180.0/np.pi)
                return 360-ang if ang > 180 else ang
            
            if check_ang(l_h, l_k, l_a) > flag_threshold or check_ang(r_h, r_k, r_a) > flag_threshold:
                cv2.putText(image, "NICE FLAG!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        out.write(image)
        frame_count += 1
        if frame_count % 10 == 0: progress_bar.progress(min(frame_count/total_frames, 1.0))

    cap.release()
    out.release()
    progress_bar.empty()
    return output_path

# ================= UI 布局 =================
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("1. Source Video")
    if 'processed_video' in st.session_state:
        if st.button("🔄 Analyze New Video"):
            st.session_state.clear()
            st.rerun()
    uploaded_file = st.file_uploader("Upload", type=['mov', 'mp4'])

if uploaded_file:
    if 'original_video' not in st.session_state:
        t = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
        t.write(uploaded_file.read())
        st.session_state['original_video'] = t.name
    
    with col1:
        st.video(st.session_state['original_video'])
        if 'processed_video' not in st.session_state:
            if st.button("Analyze Key Moves 🚀", type="primary"):
                out_name = tempfile.NamedTemporaryFile(delete=False, suffix='.webm').name
                with col2:
                    with st.spinner('Detecting power moves and rests...'):
                        res = process_video(st.session_state['original_video'], out_name, frame_skip)
                    if res and os.path.getsize(res) > 0:
                        st.session_state['processed_video'] = res
                        st.rerun()

if 'processed_video' in st.session_state:
    with col2:
        st.subheader("2. AI Analysis")
        res_file = st.session_state['processed_video']
        with open(res_file, 'rb') as f: st.video(f.read(), format="video/webm")
        st.download_button("📥 Download Analysis", open(res_file, 'rb'), "climb_analysis.webm")
