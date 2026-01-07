# ================= UI Layout with Memory (Session State) =================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Upload")
    # 如果已经有处理结果，显示一个重置按钮
    if 'processed_video' in st.session_state:
        if st.button("🔄 Upload New Video"):
            # 清除记忆，强制重新开始
            for key in ['processed_video', 'original_video']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    uploaded_file = st.file_uploader("Upload MOV/MP4", type=['mov', 'mp4'])

if uploaded_file:
    # 记忆原始视频路径
    if 'original_video' not in st.session_state:
        suffix = os.path.splitext(uploaded_file.name)[1]
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tfile.write(uploaded_file.read())
        st.session_state['original_video'] = tfile.name
    
    with col1:
        st.video(st.session_state['original_video'])
        
        # 如果还没处理过，才显示分析按钮
        if 'processed_video' not in st.session_state:
            if st.button("Start AI Analysis 🚀", type="primary"):
                out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.webm').name
                with col2:
                    st.subheader("2. AI Result")
                    with st.spinner('Analyzing...'):
                        res = process_video(st.session_state['original_video'], out_path)
                    
                    if res and os.path.exists(res) and os.path.getsize(res) > 0:
                        st.session_state['processed_video'] = res # 关键：把结果存入记忆
                        st.rerun() # 重新运行以刷新 UI 显示结果
                    else:
                        st.error("Analysis failed.")

# 在右侧显示记忆中的结果
if 'processed_video' in st.session_state:
    with col2:
        st.subheader("2. AI Result")
        st.success("Analysis Finished (Loaded from memory)!")
        res = st.session_state['processed_video']
        with open(res, 'rb') as f:
            st.video(f.read(), format="video/webm")
        st.download_button("📥 Download WebM", open(res, 'rb'), "analysis.webm")
