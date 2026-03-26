"""
🔥 Fire Detection System — Streamlit Demo
Phát hiện đám cháy sử dụng YOLO11 (ONNX Runtime — không cần PyTorch)
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import tempfile
import os
import io
import onnxruntime as ort

# ==================== CONFIG ====================
PAGE_TITLE = "🔥 Fire Detection System"
MODEL_OPTIONS = {
    "YOLO11n — Local (CPU, nhanh)": "runs/fire_detection/weights/best_local.onnx",
    "YOLO11s — Kaggle (chính xác hơn)": "runs/fire_detection/weights/best_kaggle.onnx",
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
    """Load ONNX model với cache — không cần PyTorch"""
    if not Path(model_path).exists():
        return None
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    return session

# ==================== INFERENCE ====================
def preprocess(image_bgr: np.ndarray, input_size: int = 640):
    """Resize + normalize ảnh về tensor NCHW float32"""
    h, w = image_bgr.shape[:2]
    # Letterbox resize
    scale = min(input_size / h, input_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image_bgr, (new_w, new_h))
    pad_h = (input_size - new_h) // 2
    pad_w = (input_size - new_w) // 2
    canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    canvas[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
    # BGR → RGB, HWC → NCHW, /255
    tensor = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(tensor, 0), scale, pad_w, pad_h

def xywh2xyxy(boxes):
    """Convert cx,cy,w,h → x1,y1,x2,y2"""
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    return np.stack([x1, y1, x2, y2], axis=1)

def nms(boxes_xyxy, scores, iou_thresh):
    """Simple NMS"""
    x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thresh]
    return keep

def run_inference(session, image_bgr: np.ndarray, conf: float, iou: float):
    """Chạy ONNX detection, trả về ảnh annotated + detections"""
    orig_h, orig_w = image_bgr.shape[:2]
    input_size = 640
    tensor, scale, pad_w, pad_h = preprocess(image_bgr, input_size)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: tensor})
    # YOLO ONNX output shape: [1, 8, 8400] — (batch, 4+num_classes, anchors)
    pred = outputs[0][0]  # shape (8, 8400)
    pred = pred.T          # shape (8400, 8)

    boxes_xywh = pred[:, :4]
    class_scores = pred[:, 4:]  # shape (8400, 4)
    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(len(class_scores)), class_ids]

    # Filter by confidence
    mask = confidences >= conf
    if mask.sum() == 0:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), []

    boxes_xywh = boxes_xywh[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    # Convert to xyxy & scale back to original image
    boxes_xyxy = xywh2xyxy(boxes_xywh)
    boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - pad_w) / scale
    boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - pad_h) / scale
    boxes_xyxy = np.clip(boxes_xyxy, 0, [orig_w, orig_h, orig_w, orig_h])

    # NMS
    keep = nms(boxes_xyxy, confidences, iou)
    boxes_xyxy = boxes_xyxy[keep]
    confidences = confidences[keep]
    class_ids = class_ids[keep]

    # Draw annotations
    annotated = image_bgr.copy()
    detections = []
    for i, (box, conf_val, cls_id) in enumerate(zip(boxes_xyxy, confidences, class_ids)):
        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
        color_rgb = CLASS_COLORS_RGB.get(cls_name, (200, 200, 200))
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, 2)
        label = f"{cls_name} {conf_val:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color_bgr, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        detections.append({'class': cls_name, 'confidence': float(conf_val), 'bbox': box.tolist()})

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
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

    # State để lưu ảnh đã chọn
    image_np_bgr = None
    image_pil_display = None

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
        sample_images = sorted(sample_dir.glob("*.jpg"))[:6] if sample_dir.exists() else []

        selected_sample = None
        if sample_images:
            sample_names = [p.name[:35] for p in sample_images]
            selected_sample = st.selectbox("Chọn ảnh mẫu", ["— Không dùng —"] + sample_names)

        # Ưu tiên: uploaded file > sample image
        if uploaded is not None:
            img_bytes = uploaded.read()
            image_pil_display = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            image_np = np.array(image_pil_display)
            image_np_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            st.image(image_pil_display, caption="Ảnh gốc", use_container_width=True)
        elif selected_sample and selected_sample != "— Không dùng —":
            idx = sample_names.index(selected_sample)
            with open(sample_images[idx], "rb") as f:
                img_bytes = f.read()
            image_pil_display = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            image_np = np.array(image_pil_display)
            image_np_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            st.image(image_pil_display, caption="Ảnh mẫu", use_container_width=True)

    with col2:
        st.markdown("#### 🔍 Kết quả Detection")

        if image_np_bgr is not None:
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
                st.exception(e)
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
