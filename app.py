import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os

# ================= 🛠️ 稳定导入 =================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ================= 页面配置 =================
st.set_page_config(page_title="Climbing AI Coach", page_icon="🧗", layout="wide")
st.title("🧗 AI Climbing Coach (High Performance)")

# ================= 侧边栏设置 =================
with st.sidebar:
    st.header("🔧 Performance Settings")
    
    # 核心加速开关：跳帧处理
    # Standard: 逐帧分析 (慢)
    # Fast: 每 3 帧分析 1 帧 (推荐)
    # Turbo: 每 5 帧分析 1 帧 (极快)
    processing_speed = st.select_slider(
        "Processing Speed (Frame Skipping)",
        options=["Standard", "Fast", "Turbo"],
        value="Fast"
    )
    speed_map = {"Standard": 0, "Fast": 2, "Turbo": 4}
    frame_skip = speed_map[processing_speed]
    
    st.divider()
    st.header("🧗 Coaching Settings")
    flag_threshold = st.slider("Flagging Angle Threshold", 130, 170, 150)
    show_skeleton = st.checkbox("Show Skeleton", value=True)
    show_trail = st.checkbox("Show Hip Trajectory", value=True)

# ================= 核心分析逻辑 =================
def process_video(input_path, output_path, skip_count):
    # 使用默认模型复杂度 (不使用 Lite)，平衡精度与稳定性
    pose = mp_pose.Pose(
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(input_path)
    # 获取原始视频参数
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    
    # 策略：如果视频分辨率过高（如 4K/1080P），将其等比例缩放到 720p 进行处理
    # 这能显著减少 CPU 负担，而不损失 AI 识别率
    target_h = 720
    if orig_h > target_h:
        scale = target_h / orig_h
        width = int(orig_w * scale)
        height = target_h
    else:
        width, height = orig_w, orig_h

    # 使用 VP80 编码生成 WebM 文件
    fourcc = cv2.VideoWriter_fourcc(*'VP80') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    hip_trail = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # 尺寸预处理
        if orig_h > target_h:
            frame = cv2.resize(frame, (width, height))

        # --- 性能优化：跳帧判断 ---
        # 如果不是目标帧，则跳过 AI 计算，直接写入原始画面
        if frame_count % (skip_count + 1) != 0:
            out.write(frame)
            frame_count += 1
            continue

        # --- AI 计算部分 ---
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            def get_pt(idx): return [lm[idx].x * width, lm[idx].y * height]
            
            # 轨迹追踪
            if show_trail:
                hip = get_pt(23)
                hip_trail.append((int(hip[0]), int(hip[1])))
                if len(hip_trail) > 50: hip_trail.pop(0)
                for i in range(1, len(hip_trail)):
                    cv2.line(image, hip_trail[i-1], hip_trail[i], (0, 255, 255), 2)

            # Flagging 判定逻辑
            l_h, l_k, l_a = get_pt(23), get_pt(25), get_pt(27)
            r_h, r_k, r_a = get_pt(24), get_pt(26), get_pt(28)
            
            def check_ang(a, b, c):
                a, b, c = np.array(a), np.array(b), np.array(c)
                rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
                ang = np.abs(rad*180.0/np.pi)
                return 360-ang if ang > 180 else ang

            if check_ang(l_h, l_k, l_a) > flag_threshold or check_ang(r_h, r_k, r_a) > flag_threshold:
                cv2.putText(image, "NICE FLAG!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            # 骨架绘制
            if show_skeleton:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        out.write(image)
        frame_count += 1
        
        # 每 10 帧更新一次进度条，节省 UI 刷新开销
        if frame_count % 10 == 0:
            progress_bar.progress(min(frame_count/total_frames, 1.0))

    cap.release()
    out.release()
    progress_bar.empty()
    return output_path

# ================= UI 布局 (Session State) =================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Source Video")
    if 'processed_video' in st.session_state:
        if st.button("🔄 Analyze New Video"):
            st.session_state.clear()
            st.rerun()

    uploaded_file = st.file_uploader("Upload Climbing Video (MOV/MP4)", type=['mov', 'mp4'])

if uploaded_file:
    if 'original_video' not in st.session_state:
        t = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(
