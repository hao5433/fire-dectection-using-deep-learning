"""
Các hàm tiện ích cho dự án phát hiện đám cháy
Utility Functions for Fire Detection
"""

import os
import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict
import config

def load_yaml(yaml_path: str) -> dict:
    """Load file YAML"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_yaml(data: dict, yaml_path: str):
    """Lưu dictionary sang file YAML"""
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, sort_keys=False)

def count_images(directory: Path, extensions: List[str] = None) -> int:
    """Đếm số lượng ảnh trong thư mục"""
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    count = 0
    for ext in extensions:
        count += len(list(directory.rglob(f'*{ext}')))
    return count

def count_labels(directory: Path) -> int:
    """Đếm số lượng file label (.txt)"""
    return len(list(directory.rglob('*.txt')))

def get_dataset_stats() -> Dict[str, int]:
    """Lấy thống kê dataset"""
    stats = {}
    
    for split in ['train', 'valid', 'test']:
        split_dir = config.DATASET_ROOT / split / 'images'
        if split_dir.exists():
            stats[split] = count_images(split_dir)
        else:
            stats[split] = 0
    
    return stats

def draw_boxes(image: np.ndarray, boxes: np.ndarray, class_ids: np.ndarray, 
               confidences: np.ndarray = None, class_names: List[str] = None) -> np.ndarray:
    """
    Vẽ bounding boxes lên ảnh
    
    Args:
        image: Ảnh input (numpy array)
        boxes: Array of boxes [[x1, y1, x2, y2], ...]
        class_ids: Array of class IDs
        confidences: Array of confidence scores
        class_names: List tên classes
    
    Returns:
        Ảnh đã vẽ boxes
    """
    if class_names is None:
        class_names = config.CLASS_NAMES
    
    img = image.copy()
    h, w = img.shape[:2]
    
    for idx, (box, cls_id) in enumerate(zip(boxes, class_ids)):
        cls_id = int(cls_id)
        cls_name = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
        color = config.CLASS_COLORS.get(cls_name, (255, 255, 255))
        
        # Vẽ box
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Tạo label
        if confidences is not None:
            label = f"{cls_name}: {confidences[idx]:.2f}"
        else:
            label = cls_name
        
        # Vẽ label background
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        
        # Vẽ label text
        cv2.putText(img, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return img

def yolo_to_xyxy(boxes: np.ndarray, img_width: int, img_height: int) -> np.ndarray:
    """
    Chuyển đổi YOLO format (x_center, y_center, w, h) sang (x1, y1, x2, y2)
    
    Args:
        boxes: Array of boxes in YOLO format (normalized 0-1)
        img_width: Chiều rộng ảnh
        img_height: Chiều cao ảnh
    
    Returns:
        Boxes in xyxy format
    """
    boxes_xyxy = boxes.copy()
    
    # Convert center to corners
    boxes_xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * img_width   # x1
    boxes_xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * img_height  # y1
    boxes_xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * img_width   # x2
    boxes_xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * img_height  # y2
    
    return boxes_xyxy

def load_yolo_annotation(label_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Đọc file annotation YOLO format
    
    Returns:
        class_ids: Array of class IDs
        boxes: Array of boxes in YOLO format (x_center, y_center, w, h)
    """
    if not os.path.exists(label_path):
        return np.array([]), np.array([])
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    if not lines:
        return np.array([]), np.array([])
    
    class_ids = []
    boxes = []
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            class_ids.append(int(parts[0]))
            boxes.append([float(x) for x in parts[1:5]])
    
    return np.array(class_ids), np.array(boxes)

def visualize_sample(image_path: str, label_path: str = None, save_path: str = None):
    """Visualize một mẫu ảnh với annotations"""
    # Đọc ảnh
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    # Đọc annotations nếu có
    if label_path and os.path.exists(label_path):
        class_ids, boxes_yolo = load_yolo_annotation(label_path)
        
        if len(boxes_yolo) > 0:
            # Convert YOLO to xyxy
            boxes_xyxy = yolo_to_xyxy(boxes_yolo, w, h)
            img = draw_boxes(img, boxes_xyxy, class_ids)
    
    # Hiển thị hoặc lưu
    if save_path:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, img_bgr)
    else:
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title(Path(image_path).name)
        plt.tight_layout()
        plt.show()

def create_output_dirs():
    """Tạo các thư mục output cần thiết"""
    dirs = [
        config.RUNS_DIR,
        config.MODELS_DIR,
        config.OUTPUT_DIR,
    ]
    
    for d in dirs:
        d.mkdir(exist_ok=True, parents=True)
    
    print(f"✅ Đã tạo thư mục output tại: {config.PROJECT_ROOT}")

def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Tính IoU giữa 2 boxes (format: x1, y1, x2, y2)"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def print_training_summary(results_dir: Path):
    """In tóm tắt kết quả training"""
    # Đọc results.csv nếu có
    csv_path = results_dir / 'results.csv'
    if csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path)
        
        print("\n" + "=" * 60)
        print("TÓM TẮT KẾT QUẢ TRAINING")
        print("=" * 60)
        
        # Lấy metrics cuối cùng
        last_row = df.iloc[-1]
        
        metrics = [
            ('Epoch', int(last_row.get('epoch', 0))),
            ('mAP50', last_row.get('metrics/mAP50(B)', 0)),
            ('mAP50-95', last_row.get('metrics/mAP50-95(B)', 0)),
            ('Precision', last_row.get('metrics/precision(B)', 0)),
            ('Recall', last_row.get('metrics/recall(B)', 0)),
        ]
        
        for name, value in metrics:
            if isinstance(value, float):
                print(f"{name:15s}: {value:.4f}")
            else:
                print(f"{name:15s}: {value}")
        
        print("=" * 60)

if __name__ == '__main__':
    # Test functions
    print("Dataset Statistics:")
    stats = get_dataset_stats()
    for split, count in stats.items():
        print(f"  {split:10s}: {count:5d} images")
