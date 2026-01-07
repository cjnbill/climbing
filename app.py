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
                current_hip = (int(l_hip[0]), int(l_hip[1]))
                hip_trail.append(current_hip)
                if len(hip_trail) > 100: hip_trail.pop(0)
                for i in range(1, len(hip_trail)):
                    cv2.line(image, hip_trail[i-1], hip_trail[i], (0, 255, 255), 2)
                cv2.circle(image, current_hip, 6, (0, 0, 255), -1)

            # 2. Flagging Detection
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
                cv2.putText(image, f"NICE FLAG! ({disp})", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            # 3. Skeleton
            if show_skeleton:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        out.write(image)
        frame_count += 1
        
        # Update UI every 15 frames to keep CPU focused on processing
        if frame_count % 15 == 0:
            progress_percent = min(frame_count/total_frames, 1.0)
            progress_bar.progress(progress_percent)
            status_text.text(f"Processing Frame {frame_count}/{total_frames}...")

    cap.release()
    out.release()
    progress_bar.empty()
    status_text.empty()
    return output_path

# ================= UI Layout with Memory (Session State) =================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Video Source")
    
    # Option to reset and start over
    if 'processed_video' in st.session_state:
        if st.button("🔄 Analyze New Video"):
            for key in ['processed_video', 'original_video']:
                if key in st.session_state: del st.session_state[key]
            st.rerun()

    uploaded_file = st.file_uploader("Upload MOV or MP4 (Max 20s recommended)", type=['mov', 'mp4'])

if uploaded_file:
    # Persistent storage of original file to prevent re-uploading on UI change
    if 'original_video' not in st.session_state:
        suffix = os.path.splitext(uploaded_file.name)[1]
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tfile.write(uploaded_file.read())
        st.session_state['original_video'] = tfile.name
    
    with col1:
        st.video(st.session_state['original_video'])
        
        # Only show the button if we haven't processed yet
        if 'processed_video' not in st.session_state:
            if st.button("Start Turbo Analysis 🚀", type="primary"):
                out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.webm').name
                with col2:
                    st.subheader("2. AI Analysis")
                    with st.spinner('Analyzing... Speed depends on your settings.'):
                        res = process_video(st.session_state['original_video'], out_path, frame_skip)
                    
                    if res and os.path.exists(res) and os.path.getsize(res) > 0:
                        st.session_state['processed_video'] = res
                        st.rerun()
                    else:
                        st.error("Processing failed. Try a smaller video.")

# Right side persistent result display
if 'processed_video' in st.session_state:
    with col2:
        st.subheader("2. AI Analysis")
        st.success("Analysis Ready!")
        res_path = st.session_state['processed_video']
        with open(res_path, 'rb') as f:
            st.video(f.read(), format="video/webm")
        st.download_button("📥 Download WebM Result", open(res_path, 'rb'), "climb_analysis.webm")
