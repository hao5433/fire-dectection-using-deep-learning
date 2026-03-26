"""
🔥 Fire Detection System — Streamlit Demo
YOLO11 · ONNX Runtime · PIL only (no OpenCV, no PyTorch)
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
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
    'fire':    (255, 50,  50),
    'light':   (255, 165, 0),
    'nonfire': (50,  205, 50),
    'smoke':   (160, 160, 160),
}
CLASS_EMOJI = {
    'fire':    '🔥',
    'light':   '💡',
    'nonfire': '✅',
    'smoke':   '💨',
}

# ==================== PAGE SETUP ====================
st.set_page_config(page_title=PAGE_TITLE, page_icon="🔥", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main-title { font-size:2.5rem; font-weight:700; text-align:center; color:#FF4B4B; margin-bottom:0.2rem; }
    .sub-title  { font-size:1rem; text-align:center; color:#888; margin-bottom:2rem; }
    .alert-fire {
        background:linear-gradient(135deg,#FF4B4B22,#FF4B4B44);
        border:2px solid #FF4B4B; border-radius:10px; padding:1rem;
        text-align:center; font-size:1.3rem; font-weight:bold; color:#FF4B4B;
    }
    .alert-safe {
        background:linear-gradient(135deg,#00C85122,#00C85144);
        border:2px solid #00C851; border-radius:10px; padding:1rem;
        text-align:center; font-size:1.3rem; font-weight:bold; color:#00C851;
    }
    div[data-testid="stImage"] img { border-radius:10px; }
</style>
""", unsafe_allow_html=True)

# ==================== MODEL LOADER ====================
@st.cache_resource
def load_model(model_path: str):
    if not Path(model_path).exists():
        return None
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])

# ==================== PREPROCESS (PIL only, no cv2) ====================
def letterbox_pil(pil_img: Image.Image, size: int = 640):
    w, h = pil_img.size
    scale = min(size / h, size / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = pil_img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    tensor = np.array(canvas, dtype=np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0
    return tensor, scale, pad_x, pad_y

# ==================== NMS ====================
def nms(boxes, scores, iou_thresh):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        inter = (np.maximum(0, np.minimum(x2[i], x2[order[1:]]) - np.maximum(x1[i], x1[order[1:]])) *
                 np.maximum(0, np.minimum(y2[i], y2[order[1:]]) - np.maximum(y1[i], y1[order[1:]])))
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thresh]
    return keep

# ==================== INFERENCE ====================
def run_inference(session, pil_img: Image.Image, conf: float, iou: float):
    orig_w, orig_h = pil_img.size
    tensor, scale, pad_x, pad_y = letterbox_pil(pil_img)

    pred = session.run(None, {session.get_inputs()[0].name: tensor})[0][0].T  # (8400, 8)
    cls_scores = pred[:, 4:]
    cls_ids    = np.argmax(cls_scores, axis=1)
    confs      = cls_scores[np.arange(len(cls_scores)), cls_ids]

    mask = confs >= conf
    if mask.sum() == 0:
        return pil_img.copy(), []

    pred, confs, cls_ids = pred[mask], confs[mask], cls_ids[mask]
    cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    x1 = np.clip((cx - bw / 2 - pad_x) / scale, 0, orig_w)
    y1 = np.clip((cy - bh / 2 - pad_y) / scale, 0, orig_h)
    x2 = np.clip((cx + bw / 2 - pad_x) / scale, 0, orig_w)
    y2 = np.clip((cy + bh / 2 - pad_y) / scale, 0, orig_h)
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    keep   = nms(boxes, confs, iou)
    boxes  = boxes[keep]; confs = confs[keep]; cls_ids = cls_ids[keep]

    # Draw với PIL — không cần cv2
    annotated = pil_img.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    detections = []
    for box, cf, cid in zip(boxes, confs, cls_ids):
        cls_name = CLASS_NAMES[int(cid)] if int(cid) < len(CLASS_NAMES) else f"cls{cid}"
        color    = CLASS_COLORS_RGB.get(cls_name, (200, 200, 200))
        x1_, y1_, x2_, y2_ = [float(v) for v in box]
        draw.rectangle([x1_, y1_, x2_, y2_], outline=color, width=3)
        label = f"{cls_name} {cf:.2f}"
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        draw.rectangle([x1_, max(0, y1_ - th - 6), x1_ + tw + 6, y1_], fill=color)
        draw.text((x1_ + 3, max(0, y1_ - th - 4)), label, fill=(255, 255, 255), font=font)
        detections.append({'class': cls_name, 'confidence': float(cf), 'bbox': box.tolist()})

    return annotated, detections

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("## ⚙️ Cài đặt"); st.divider()
    st.markdown("**🤖 Model**")
    selected_model_name = st.selectbox("Model", list(MODEL_OPTIONS.keys()), label_visibility="collapsed")
    model_path = MODEL_OPTIONS[selected_model_name]

    st.markdown("**🎯 Confidence Threshold**")
    conf_threshold = st.slider("Conf", 0.05, 0.95, 0.15, 0.05, label_visibility="collapsed",
                                help="Càng thấp → detect nhiều hơn. Càng cao → chắc chắn hơn.")
    st.markdown("**📐 IoU Threshold (NMS)**")
    iou_threshold = st.slider("IoU", 0.1, 0.9, 0.45, 0.05, label_visibility="collapsed",
                               help="Ngưỡng loại bỏ bounding box trùng lắp.")
    st.divider()
    st.markdown("### 📊 Thông tin Model")
    st.markdown("| | |\n|---|---|\n| **mAP50** | 0.711 |\n| **Fire Recall** | 77.9% |\n| **Classes** | 4 |\n| **Framework** | YOLO11 |")
    st.divider()
    st.markdown("### 🏷️ Classes")
    for cls in CLASS_NAMES:
        color = CLASS_COLORS_RGB[cls]
        st.markdown(f'<span style="color:rgb{color}">■</span> {CLASS_EMOJI[cls]} **{cls}**', unsafe_allow_html=True)

# ==================== HEADER ====================
st.markdown('<div class="main-title">🔥 Fire Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Phát hiện đám cháy sử dụng YOLO11 · ONNX Runtime · PTIT · 2026</div>', unsafe_allow_html=True)

model = load_model(model_path)
if model is None:
    st.error(f"❌ Không tìm thấy model: `{model_path}`")
    st.stop()
else:
    st.success(f"✅ Model loaded: **{selected_model_name}**")
st.divider()

# ==================== TABS ====================
tab1, tab2 = st.tabs(["🖼️ Ảnh tĩnh", "ℹ️ Hướng dẫn"])

with tab1:
    col1, col2 = st.columns(2, gap="large")
    pil_input = None

    with col1:
        st.markdown("#### 📤 Upload ảnh")
        uploaded = st.file_uploader("Ảnh (JPG/PNG)", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

        st.markdown("**Hoặc dùng ảnh mẫu:**")
        sample_dir = Path("output/kaggle_results")
        sample_images = sorted(sample_dir.glob("*.jpg"))[:6] if sample_dir.exists() else []
        selected_sample = None
        if sample_images:
            sample_names = [p.name[:40] for p in sample_images]
            selected_sample = st.selectbox("Ảnh mẫu", ["— Không dùng —"] + sample_names)

        if uploaded is not None:
            pil_input = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
            st.image(pil_input, caption="Ảnh gốc", use_container_width=True)
        elif selected_sample and selected_sample != "— Không dùng —":
            idx = sample_names.index(selected_sample)
            pil_input = Image.open(sample_images[idx]).convert("RGB")
            st.image(pil_input, caption="Ảnh mẫu", use_container_width=True)

    with col2:
        st.markdown("#### 🔍 Kết quả Detection")
        if pil_input is not None:
            try:
                with st.spinner("Đang phân tích..."):
                    annotated, detections = run_inference(model, pil_input, conf_threshold, iou_threshold)
                st.image(annotated, caption="Kết quả Detection", use_container_width=True)

                has_fire  = any(d['class'] == 'fire'  for d in detections)
                has_smoke = any(d['class'] == 'smoke' for d in detections)
                if has_fire:
                    st.markdown('<div class="alert-fire">🚨 PHÁT HIỆN ĐÁM CHÁY!</div>', unsafe_allow_html=True)
                elif has_smoke:
                    st.warning("💨 Phát hiện **khói** — Cảnh báo sớm!")
                elif detections:
                    st.markdown('<div class="alert-safe">✅ Không có lửa</div>', unsafe_allow_html=True)
                else:
                    st.info("ℹ️ Không phát hiện gì (thử giảm Confidence Threshold)")

                if detections:
                    st.markdown("**📋 Chi tiết:**")
                    for d in sorted(detections, key=lambda x: x['confidence'], reverse=True):
                        emoji = CLASS_EMOJI.get(d['class'], '❓')
                        color = CLASS_COLORS_RGB.get(d['class'], (200, 200, 200))
                        st.markdown(
                            f'<span style="color:rgb{color}">{emoji} **{d["class"].upper()}**</span>'
                            f' — confidence: **{d["confidence"]:.1%}**',
                            unsafe_allow_html=True
                        )
            except Exception as e:
                st.error(f"Lỗi inference: {e}")
                st.exception(e)
        else:
            st.info("👆 Upload ảnh hoặc chọn ảnh mẫu để bắt đầu")

with tab2:
    st.markdown("""
    ### 🚀 Cách sử dụng
    1. **Upload ảnh** hoặc chọn ảnh mẫu từ dropdown
    2. Điều chỉnh **Confidence Threshold** ở sidebar (mặc định 0.15)
    3. Xem kết quả detection ở cột bên phải

    ### 🏷️ Các class được detect
    | Class | Ý nghĩa | Màu |
    |-------|---------|-----|
    | 🔥 fire | Đám lửa | Đỏ |
    | 💡 light | Ánh sáng (false positive) | Cam |
    | ✅ nonfire | Không có lửa | Xanh lá |
    | 💨 smoke | Khói | Xám |

    ### 📊 Thông số model
    - **Architecture:** YOLO11 (Ultralytics)
    - **Format:** ONNX (không cần PyTorch)
    - **mAP@50:** 0.711 · **Fire Recall:** 77.9%
    - **Training:** Kaggle T4 GPU + Local CPU
    """)
