"""
Cấu hình cho dự án phát hiện đám cháy
Fire Detection Configuration
"""

from pathlib import Path

# ==================== ĐƯỜNG DẪN ====================
PROJECT_ROOT = Path(__file__).parent
DATASET_ROOT = PROJECT_ROOT
DATA_YAML = PROJECT_ROOT / 'data.yaml'

# Thư mục train/valid/test
TRAIN_DIR = DATASET_ROOT / 'train'
VALID_DIR = DATASET_ROOT / 'valid'
TEST_DIR = DATASET_ROOT / 'test'

# Thư mục lưu kết quả
RUNS_DIR = PROJECT_ROOT / 'runs'
MODELS_DIR = PROJECT_ROOT / 'models'
OUTPUT_DIR = PROJECT_ROOT / 'output'

# ==================== DATASET ====================
CLASS_NAMES = ['fire', 'light', 'nonfire', 'smoke']
NUM_CLASSES = 4

# Màu sắc cho từng class (BGR format cho OpenCV)
CLASS_COLORS = {
    'fire':    (0, 0, 255),      # Đỏ
    'light':   (0, 165, 255),    # Cam
    'nonfire': (0, 255, 0),      # Xanh lá
    'smoke':   (128, 128, 128),  # Xám
}

# ==================== MODEL ====================
# YOLOv11 model sizes: n (nano), s (small), m (medium), l (large), x (xlarge)
MODEL_SIZE = 'yolo11n'  # Thay đổi thành 's', 'm', 'l', 'x' nếu cần
MODEL_NAME = f'{MODEL_SIZE}.pt'

# ==================== TRAINING ====================
EPOCHS = 100
BATCH_SIZE = 16
IMAGE_SIZE = 640
DEVICE = 0  # 0 = GPU đầu tiên, 'cpu' = CPU

# Learning rate & optimizer
LEARNING_RATE = 0.01
OPTIMIZER = 'SGD'  # 'SGD', 'Adam', 'AdamW'

# Early stopping
PATIENCE = 20  # Dừng sớm nếu không cải thiện sau N epochs

# Data augmentation
AUGMENT = True
MOSAIC = 1.0  # Mosaic augmentation probability
MIXUP = 0.1   # Mixup augmentation probability

# ==================== INFERENCE ====================
CONFIDENCE_THRESHOLD = 0.15  # Ngưỡng confidence cho detection (giảm từ 0.25)
IOU_THRESHOLD = 0.45        # Ngưỡng IoU cho NMS
MAX_DETECTIONS = 300        # Số lượng detection tối đa

# ==================== EVALUATION ====================
EVAL_BATCH_SIZE = 16

# ==================== EXPORT ====================
EXPORT_FORMATS = ['onnx', 'torchscript']  # Format để export model

# ==================== LOGGING ====================
VERBOSE = True  # In chi tiết quá trình training
SAVE_PLOTS = True  # Lưu plots

# ==================== REAL-TIME DETECTION ====================
WEBCAM_INDEX = 0  # Index của webcam (thường là 0)
VIDEO_FPS = 30    # FPS cho video output

# ==================== ADVANCED ====================
# Multi-scale training
MULTI_SCALE = False
SCALE_RANGE = (0.5, 1.5)

# Label smoothing
LABEL_SMOOTHING = 0.0

# Class weights (nếu dataset imbalanced)
CLASS_WEIGHTS = None  # [1.0, 1.0, 1.0, 1.0]

def get_config():
    """Trả về config dưới dạng dictionary"""
    return {
        'data': str(DATA_YAML),
        'epochs': EPOCHS,
        'batch': BATCH_SIZE,
        'imgsz': IMAGE_SIZE,
        'device': DEVICE,
        'lr0': LEARNING_RATE,
        'optimizer': OPTIMIZER,
        'patience': PATIENCE,
        'augment': AUGMENT,
        'mosaic': MOSAIC,
        'mixup': MIXUP,
        'verbose': VERBOSE,
        'plots': SAVE_PLOTS,
    }

def print_config():
    """In ra cấu hình hiện tại"""
    print("=" * 60)
    print("CẤU HÌNH DỰ ÁN PHÁT HIỆN ĐÁM CHÁY")
    print("=" * 60)
    print(f"Dataset:        {DATASET_ROOT}")
    print(f"Classes:        {CLASS_NAMES}")
    print(f"Model:          {MODEL_NAME}")
    print(f"Epochs:         {EPOCHS}")
    print(f"Batch Size:     {BATCH_SIZE}")
    print(f"Image Size:     {IMAGE_SIZE}")
    print(f"Device:         {'GPU' if DEVICE == 0 else 'CPU'}")
    print(f"Learning Rate:  {LEARNING_RATE}")
    print(f"Confidence:     {CONFIDENCE_THRESHOLD}")
    print("=" * 60)

if __name__ == '__main__':
    print_config()
