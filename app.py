import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os
import time

# ================= Stable Import Check =================
try:
    import mediapipe.solutions.pose as mp_pose
    import mediapipe.solutions.drawing_utils as mp_drawing
except ImportError:
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import drawing_utils as mp_drawing

# ================= Page Config =================
st.set_page_config(page_title="Climbing AI Coach", page_icon="🧗", layout="wide")
st.title("🧗 AI Climbing Coach (MOV Compatible)")

# ================= Sidebar =================
with st.sidebar:
    st.header("🔧 Settings")
    flag_threshold = st.slider("Flagging Threshold", 130, 170, 150)
    st.divider()
    show_skeleton = st.checkbox("Show Skeleton", value=True)
    show_trail = st.checkbox("Show Hip Trajectory", value=True)
    st.info("Note: MOV files will be processed and exported as MP4.")

# ================= Core Analysis Logic =================
def process_video(input_path, output_path):
    # Initialize MediaPipe
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise Exception("Could not open video file. The format might be unsupported.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 30
    
    # Use 'avc1' or 'H264' for MP4 containers on Cloud Linux
    # If this fails, the error will be caught in the main block
    fourcc = cv2.VideoWriter_fourcc(*'H264') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    hip_trail = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Progress UI
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    frame_count = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Process frame
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

            out.write(image)
            frame_count += 1
            if total_frames > 0:
                percent = min(frame_count/total_frames, 1.0)
                progress_bar.progress(percent)
                status_text.text(f"Processing frame {frame_count}/{total_frames} ({int(percent*100)}%)")
                
    except Exception as e:
        st.error(f"Error during frame processing: {e}")
    finally:
        cap.release()
        out.release()
        progress_bar.empty()
        status_text.empty()
        
    return output_path

# ================= UI Layout =================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Upload Video")
    uploaded_file = st.file_uploader("Upload MOV or MP4", type=['mov', 'mp4'])

if uploaded_file:
    # Save original to temp
    suffix = os.path.splitext(uploaded_file.name)[1]
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tfile.write(uploaded_file.read())
    
    with col1:
        st.info(f"Original Video ({suffix.upper()})")
        st.video(tfile.name)
        analyze_btn = st.button("Start Analysis 🚀", type="primary")

    if analyze_btn:
        # Output is always .mp4 for web compatibility
        out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        
        with col2:
            st.subheader("2. Result")
            try:
                with st.spinner('AI is calculating...'):
                    res = process_video(tfile.name, out_path)
                
                if os.path.exists(res) and os.path.getsize(res) > 0:
                    st.success("Analysis Complete!")
                    with open(res, 'rb') as f:
                        st.video(f.read())
                    st.download_button("📥 Download Analysis", open(res, 'rb'), file_name="analysis.mp4")
                else:
                    st.error("The output video file is empty. This usually happens due to a codec mismatch on the server.")
                    st.warning("Try converting your video to a standard 720p MP4 before uploading.")
            
            except Exception as e:
                st.error(f"Critical System Error: {e}")
                st.info("Details: Check if the video resolution is too high (try 720p) or if the file is corrupted.")
