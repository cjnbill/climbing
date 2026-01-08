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
st.set_page_config(page_title="Climbing AI Coach Pro", page_icon="🧗", layout="wide")
st.title("🧗 AI Climbing Coach (Stability & Force Analysis)")

# ================= 侧边栏设置 =================
with st.sidebar:
    st.header("🔧 Settings")
    processing_speed = st.select_slider("Speed", options=["Standard", "Fast", "Turbo"], value="Fast")
    speed_map = {"Standard": 0, "Fast": 2, "Turbo": 4}
    frame_skip = speed_map[processing_speed]
    
    st.divider()
    st.header("⚖️ Balance Analysis")
    show_balance = st.checkbox("Identify Redundant Limbs", value=True)
    flag_threshold = st.slider("Flagging Threshold", 130, 170, 150)

# ================= 几何计算辅助 =================
def point_in_triangle(p, a, b, c):
    """判定点 P 是否在三角形 ABC 内"""
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    
    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)

# ================= 核心分析逻辑 =================
def process_video(input_path, output_path, skip_count):
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(input_path)
    
    orig_w, orig_h = int(cap.get(3)), int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    target_h = 720
    scale = target_h / orig_h
    width, height = int(orig_w * scale), target_h

    fourcc = cv2.VideoWriter_fourcc(*'VP80') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (width, height))
        
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
            
            # 获取 4 个末端点
            limbs = {
                "L-Hand": get_pt(15), "R-Hand": get_pt(16),
                "L-Foot": get_pt(27), "R-Foot": get_pt(28)
            }
            # 获取重心 (Hip Center)
            hip_c = (get_pt(23) + get_pt(24)) / 2

            # --- 受力/平衡分析 ---
            if show_balance:
                names = list(limbs.keys())
                redundant_limb = None
                
                # 尝试去掉每一个肢体，检查重心是否仍在剩余三个构成的三角形内
                for i in range(4):
                    others = [limbs[names[j]] for j in range(4) if i != j]
                    if point_in_triangle(hip_c, others[0], others[1], others[2]):
                        redundant_limb = names[i]
                        break # 找到第一个冗余点就跳出
                
                # 视觉标记
                for name, pt in limbs.items():
                    color = (0, 0, 255) if name == redundant_limb else (0, 255, 0)
                    size = 5 if name == redundant_limb else 10
                    cv2.circle(image, (int(pt[0]), int(pt[1])), size, color, -1)
                    if name == redundant_limb:
                        cv2.putText(image, "REDUNDANT", (int(pt[0]), int(pt[1])-20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # 绘制重心投影
            cv2.circle(image, (int(hip_c[0]), int(hip_c[1])), 8, (255, 255, 255), 2)
            
            # 绘制骨架
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        out.write(image)
        frame_count += 1
        if frame_count % 15 == 0: progress_bar.progress(min(frame_count/total_frames, 1.0))

    cap.release()
    out.release()
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
            if st.button("Run Balance Analysis ⚖️", type="primary"):
                out_name = tempfile.NamedTemporaryFile(delete=False, suffix='.webm').name
                with col2:
                    with st.spinner('Calculating Support Polygon...'):
                        res = process_video(st.session_state['original_video'], out_name, frame_skip)
                    if res and os.path.getsize(res) > 0:
                        st.session_state['processed_video'] = res
                        st.rerun()

if 'processed_video' in st.session_state:
    with col2:
        st.subheader("2. Stability Insights")
        res_file = st.session_state['processed_video']
        with open(res_file, 'rb') as f: st.video(f.read(), format="video/webm")
        st.info("🔴 Red circles mark 'Redundant' limbs. Your COG is stable without them.")
