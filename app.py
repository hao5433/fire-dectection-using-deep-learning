"""
🔥 Fire Detection System — Streamlit Demo
Phát hiện đám cháy sử dụng YOLO11
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
import tempfile
import os

# ==================== CONFIG ====================
PAGE_TITLE = "🔥 Fire Detection System"
MODEL_OPTIONS = {
    "YOLO11n — Local (CPU, nhanh)": "runs/fire_detection/weights/best_local.pt",
    "YOLO11s — Kaggle (chính xác hơn)": "runs/fire_detection/weights/best_kaggle.pt",
}
CLASS_NAMES = ['fire', 'light', 'nonfire', 'smoke']
CLASS_COLORS_RGB = {
    'fire':    (255, 50,  50),   # Đỏ
    'light':   (255, 165, 0),    # Cam
    'nonfire': (50,  205, 50),   # Xanh lá
    'smoke':   (160, 160, 160),  # Xám
}
CLASS_EMOJI = {
    'fire':    '🔥',
    'light':   '💡',
    'nonfire': '✅',
    'smoke':   '💨',
}

# ==================== PAGE SETUP ====================
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== CSS ====================
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        color: #FF4B4B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
    }
    .alert-fire {
        background: linear-gradient(135deg, #FF4B4B22, #FF4B4B44);
        border: 2px solid #FF4B4B;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
        color: #FF4B4B;
        animation: pulse 1s infinite;
    }
    .alert-safe {
        background: linear-gradient(135deg, #00C85122, #00C85144);
        border: 2px solid #00C851;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
        color: #00C851;
    }
    .metric-card {
        background: #1E1E2E;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    div[data-testid="stImage"] img {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== MODEL LOADER ====================
@st.cache_resource
def load_model(model_path: str):
    """Load YOLO model với cache để không reload mỗi lần"""
    if not Path(model_path).exists():
        return None
    return YOLO(model_path)

# ==================== INFERENCE ====================
def run_inference(model, image: np.ndarray, conf: float, iou: float):
    """Chạy detection và trả về ảnh đã annotate + detections"""
    results = model.predict(
        source=image,
        conf=conf,
        iou=iou,
        verbose=False,
        augment=True,
    )
    result = results[0]
    annotated = result.plot()  # BGR numpy array
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
        conf_val = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()
        detections.append({
            'class': cls_name,
            'confidence': conf_val,
            'bbox': xyxy,
        })

    return annotated_rgb, detections

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("## ⚙️ Cài đặt")
    st.divider()

    # Chọn model
    st.markdown("**🤖 Model**")
    selected_model_name = st.selectbox(
        "Chọn model",
        options=list(MODEL_OPTIONS.keys()),
        label_visibility="collapsed"
    )
    model_path = MODEL_OPTIONS[selected_model_name]

    # Confidence threshold
    st.markdown("**🎯 Confidence Threshold**")
    conf_threshold = st.slider(
        "Confidence", 0.05, 0.95, 0.15, 0.05,
        help="Càng thấp → detect nhiều hơn nhưng dễ nhầm. Càng cao → chắc chắn hơn nhưng có thể bỏ sót.",
        label_visibility="collapsed"
    )

    # IoU threshold
    st.markdown("**📐 IoU Threshold (NMS)**")
    iou_threshold = st.slider(
        "IoU", 0.1, 0.9, 0.45, 0.05,
        help="Ngưỡng loại bỏ bounding box trùng lắp.",
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("### 📊 Thông tin Model")
    st.markdown(f"""
    | | |
    |---|---|
    | **mAP50** | 0.711 |
    | **Fire Recall** | 77.9% |
    | **Classes** | 4 |
    | **Framework** | YOLO11 |
    """)

    st.divider()
    st.markdown("### 🏷️ Classes")
    for cls in CLASS_NAMES:
        color = CLASS_COLORS_RGB[cls]
        emoji = CLASS_EMOJI[cls]
        st.markdown(
            f'<span style="color:rgb{color}">■</span> {emoji} **{cls}**',
            unsafe_allow_html=True
        )

# ==================== MAIN CONTENT ====================
st.markdown('<div class="main-title">🔥 Fire Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Phát hiện đám cháy real-time sử dụng YOLO11 · PTIT · 2026</div>', unsafe_allow_html=True)

# Load model
model = load_model(model_path)
if model is None:
    st.error(f"❌ Không tìm thấy model tại: `{model_path}`\n\nHãy đảm bảo file `best.pt` tồn tại.")
    st.info("💡 **Hướng dẫn:** Đặt file `best.pt` vào đúng đường dẫn, hoặc chọn model khác ở sidebar.")
    st.stop()
else:
    st.success(f"✅ Model loaded: **{selected_model_name}**")

st.divider()

# Tabs
tab1, tab2, tab3 = st.tabs(["🖼️ Ảnh tĩnh", "🎥 Video", "📷 Webcam (Live)"])

# ==================== TAB 1: ẢNH ====================
with tab1:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("#### 📤 Upload ảnh")
        uploaded = st.file_uploader(
            "Chọn ảnh (JPG, PNG, JPEG)",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

        # Ảnh mẫu từ output/kaggle_results
        st.markdown("**Hoặc dùng ảnh mẫu từ kết quả thực nghiệm:**")
        sample_dir = Path("output/kaggle_results")
        sample_images = list(sample_dir.glob("*.jpg"))[:6] if sample_dir.exists() else []

        if sample_images:
            sample_names = [p.name[:30] + "..." for p in sample_images]
            selected_sample = st.selectbox("Chọn ảnh mẫu", ["— Không dùng —"] + sample_names)
            if selected_sample != "— Không dùng —":
                idx = sample_names.index(selected_sample)
                with open(sample_images[idx], "rb") as f:
                    uploaded = f
                    image_pil = Image.open(sample_images[idx]).convert("RGB")
                    image_np = np.array(image_pil)
                    image_np_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                    st.image(image_pil, caption="Ảnh gốc", use_container_width=True)

        if uploaded is not None and not isinstance(uploaded, Path):
            image_pil = Image.open(uploaded).convert("RGB")
            image_np = np.array(image_pil)
            image_np_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            st.image(image_pil, caption="Ảnh gốc", use_container_width=True)

    with col2:
        st.markdown("#### 🔍 Kết quả Detection")

        if 'image_np_bgr' in dir() or uploaded is not None:
            try:
                with st.spinner("Đang phân tích..."):
                    annotated_rgb, detections = run_inference(
                        model, image_np_bgr, conf_threshold, iou_threshold
                    )

                st.image(annotated_rgb, caption="Kết quả Detection", use_container_width=True)

                # Alert
                has_fire = any(d['class'] == 'fire' for d in detections)
                has_smoke = any(d['class'] == 'smoke' for d in detections)

                if has_fire:
                    st.markdown('<div class="alert-fire">🚨 PHÁT HIỆN ĐÁM CHÁY!</div>', unsafe_allow_html=True)
                elif has_smoke:
                    st.warning("💨 Phát hiện **khói** — Cảnh báo sớm!")
                elif detections:
                    st.markdown('<div class="alert-safe">✅ Không có lửa</div>', unsafe_allow_html=True)
                else:
                    st.info("ℹ️ Không phát hiện đối tượng nào (thử giảm Confidence Threshold)")

                # Chi tiết detections
                if detections:
                    st.markdown("**📋 Chi tiết:**")
                    for d in sorted(detections, key=lambda x: x['confidence'], reverse=True):
                        emoji = CLASS_EMOJI.get(d['class'], '❓')
                        color = CLASS_COLORS_RGB.get(d['class'], (200, 200, 200))
                        st.markdown(
                            f'<span style="color:rgb{color}">{emoji} **{d["class"].upper()}**</span> — confidence: **{d["confidence"]:.1%}**',
                            unsafe_allow_html=True
                        )

            except Exception as e:
                st.error(f"Lỗi inference: {e}")
        else:
            st.info("👆 Upload ảnh hoặc chọn ảnh mẫu để bắt đầu")

# ==================== TAB 2: VIDEO ====================
with tab2:
    st.markdown("#### 📤 Upload Video")
    video_file = st.file_uploader(
        "Chọn video (MP4, AVI, MOV)",
        type=["mp4", "avi", "mov"],
        label_visibility="collapsed"
    )

    if video_file:
        # Lưu video tạm
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_file.read())
            tmp_path = tmp.name

        st.video(tmp_path)

        if st.button("▶️ Chạy Detection trên Video", type="primary"):
            cap = cv2.VideoCapture(tmp_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25

            stframe = st.empty()
            progress = st.progress(0, text="Đang xử lý video...")
            status_box = st.empty()

            frame_count = 0
            fire_frames = 0
            skip = max(1, int(fps // 5))  # Xử lý 5 frame/giây

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % skip != 0:
                    continue

                annotated_rgb, detections = run_inference(model, frame, conf_threshold, iou_threshold)

                if any(d['class'] == 'fire' for d in detections):
                    fire_frames += 1
                    status_box.error(f"🚨 Frame {frame_count}: PHÁT HIỆN LỬA!")
                elif any(d['class'] == 'smoke' for d in detections):
                    status_box.warning(f"💨 Frame {frame_count}: Phát hiện khói")
                else:
                    status_box.success(f"✅ Frame {frame_count}: An toàn")

                stframe.image(annotated_rgb, channels="RGB", use_container_width=True)
                progress.progress(min(frame_count / total_frames, 1.0), text=f"Frame {frame_count}/{total_frames}")

            cap.release()
            os.unlink(tmp_path)
            progress.progress(1.0, text="✅ Hoàn tất!")
            st.success(f"🎬 Xử lý xong! Phát hiện lửa trong **{fire_frames}** frames / {frame_count} frames đã xử lý.")

# ==================== TAB 3: WEBCAM ====================
with tab3:
    st.markdown("#### 📷 Live Detection qua Webcam")
    st.info("💡 **Lưu ý:** Webcam live yêu cầu chạy Streamlit trên máy local (không hoạt động trên cloud).")

    col_w1, col_w2 = st.columns([1, 1])
    with col_w1:
        run_webcam = st.toggle("🟢 Bật Webcam", value=False)

    stframe_web = st.empty()
    status_web = st.empty()

    if run_webcam:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Không mở được webcam. Kiểm tra kết nối camera.")
        else:
            st.success("✅ Webcam đang hoạt động — Toggle OFF để dừng")
            while run_webcam:
                ret, frame = cap.read()
                if not ret:
                    break

                annotated_rgb, detections = run_inference(model, frame, conf_threshold, iou_threshold)
                stframe_web.image(annotated_rgb, channels="RGB", use_container_width=True)

                if any(d['class'] == 'fire' for d in detections):
                    status_web.error("🚨 PHÁT HIỆN LỬA!")
                elif any(d['class'] == 'smoke' for d in detections):
                    status_web.warning("💨 Phát hiện khói!")
                else:
                    status_web.success("✅ An toàn")

            cap.release()
