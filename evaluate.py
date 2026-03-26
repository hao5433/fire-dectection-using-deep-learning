"""
Đánh giá model YOLOv11
Evaluate YOLOv11 Model Performance
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
import config
import utils

def evaluate_model(model_path: str = None, save_results: bool = True):
    """
    Đánh giá model trên validation/test set
    
    Args:
        model_path: Đường dẫn đến model weights
        save_results: Lưu kết quả đánh giá
    """
    print("\n" + "=" * 70)
    print("ĐÁNH GIÁ MODEL PHÁT HIỆN ĐÁM CHÁY")
    print("=" * 70)
    
    # Load model
    if model_path is None:
        model_path = config.RUNS_DIR / 'fire_detection' / 'weights' / 'best.pt'
    
    if not Path(model_path).exists():
        print(f"❌ Không tìm thấy model: {model_path}")
        return None
    
    print(f"\n📦 Đang load model từ: {model_path}")
    model = YOLO(str(model_path))
    print("✅ Đã load model thành công!")
    
    # Evaluate
    print("\n📊 Đang đánh giá trên validation set...")
    print("-" * 70)
    
    metrics = model.val(
        data=str(config.DATA_YAML),
        batch=config.EVAL_BATCH_SIZE,
        imgsz=config.IMAGE_SIZE,
        device=config.DEVICE,
        plots=save_results,
        save_json=save_results,
    )
    
    # In kết quả
    print("\n" + "=" * 70)
    print("KẾT QUẢ ĐÁNH GIÁ")
    print("=" * 70)
    
    results = {
        'mAP50': metrics.box.map50,
        'mAP50-95': metrics.box.map,
        'Precision': metrics.box.mp,
        'Recall': metrics.box.mr,
    }
    
    for metric_name, value in results.items():
        print(f"  {metric_name:15s}: {value:.4f}")
    
    # Per-class metrics
    if hasattr(metrics.box, 'ap_class_index'):
        print("\n" + "-" * 70)
        print("PER-CLASS METRICS (mAP50):")
        print("-" * 70)
        
        for idx, cls_id in enumerate(metrics.box.ap_class_index):
            cls_name = config.CLASS_NAMES[int(cls_id)]
            ap50 = metrics.box.ap50[idx]
            print(f"  {cls_name:15s}: {ap50:.4f}")
    
    print("=" * 70)
    
    return metrics

def test_on_images(model_path: str = None, test_dir: str = None, 
                   num_samples: int = 10, save_dir: str = None):
    """
    Test model trên ảnh mẫu
    
    Args:
        model_path: Đường dẫn đến model
        test_dir: Thư mục chứa ảnh test
        num_samples: Số lượng ảnh để test
        save_dir: Thư mục lưu kết quả
    """
    print("\n🧪 Testing model trên ảnh mẫu...")
    
    # Load model
    if model_path is None:
        model_path = config.RUNS_DIR / 'fire_detection' / 'weights' / 'best.pt'
    
    if not Path(model_path).exists():
        print(f"❌ Không tìm thấy model: {model_path}")
        return
    
    model = YOLO(str(model_path))
    
    # Lấy ảnh test
    if test_dir is None:
        test_dir = config.TEST_DIR / 'images'
        if not test_dir.exists():
            test_dir = config.VALID_DIR / 'images'
    
    test_images = list(Path(test_dir).glob('*.jpg'))[:num_samples]
    
    if not test_images:
        print(f"❌ Không tìm thấy ảnh trong: {test_dir}")
        return
    
    # Predict
    print(f"\n🔍 Testing trên {len(test_images)} ảnh...")
    
    rows = (len(test_images) + 2) // 3
    cols = min(3, len(test_images))
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if len(test_images) == 1:
        axes = [axes]
    else:
        axes = axes.ravel()
    
    for idx, img_path in enumerate(test_images):
        # Predict
        results = model.predict(
            source=str(img_path),
            conf=config.CONFIDENCE_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            verbose=False
        )
        
        # Visualize
        annotated = results[0].plot()
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
        axes[idx].imshow(annotated)
        axes[idx].set_title(img_path.name, fontsize=10)
        axes[idx].axis('off')
        
        # Print detections
        boxes = results[0].boxes
        print(f"\n  {img_path.name}: {len(boxes)} detections")
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = config.CLASS_NAMES[cls_id]
            print(f"    - {cls_name}: {conf:.2%}")
    
    # Ẩn axes thừa
    for idx in range(len(test_images), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'test_results.png'
        save_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Đã lưu kết quả tại: {save_path}")
    else:
        plt.show()

def confusion_matrix_analysis(model_path: str = None):
    """Phân tích confusion matrix"""
    print("\n📊 Phân tích Confusion Matrix...")
    
    # Load model
    if model_path is None:
        model_path = config.RUNS_DIR / 'fire_detection' / 'weights' / 'best.pt'
    
    if not Path(model_path).exists():
        print(f"❌ Không tìm thấy model: {model_path}")
        return
    
    model = YOLO(str(model_path))
    
    # Validate để tạo confusion matrix
    metrics = model.val(
        data=str(config.DATA_YAML),
        plots=True,
        save_json=True
    )
    
    # Confusion matrix được lưu tự động trong runs/detect/val
    print("✅ Confusion matrix đã được tạo và lưu")

def compare_models(model_paths: list, model_names: list = None):
    """
    So sánh nhiều models
    
    Args:
        model_paths: List đường dẫn đến các models
        model_names: List tên models (để hiển thị)
    """
    print("\n" + "=" * 70)
    print("SO SÁNH MODELS")
    print("=" * 70)
    
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(model_paths))]
    
    results_list = []
    
    for model_path, model_name in zip(model_paths, model_names):
        if not Path(model_path).exists():
            print(f"⚠️  Bỏ qua {model_name}: không tìm thấy file")
            continue
        
        print(f"\n📦 Đánh giá {model_name}...")
        model = YOLO(str(model_path))
        
        metrics = model.val(
            data=str(config.DATA_YAML),
            batch=config.EVAL_BATCH_SIZE,
            verbose=False
        )
        
        results_list.append({
            'name': model_name,
            'mAP50': metrics.box.map50,
            'mAP50-95': metrics.box.map,
            'Precision': metrics.box.mp,
            'Recall': metrics.box.mr,
        })
    
    # In bảng so sánh
    if results_list:
        print("\n" + "=" * 70)
        print(f"{'Model':<20s} {'mAP50':>10s} {'mAP50-95':>10s} {'Precision':>10s} {'Recall':>10s}")
        print("-" * 70)
        
        for result in results_list:
            print(f"{result['name']:<20s} "
                  f"{result['mAP50']:>10.4f} "
                  f"{result['mAP50-95']:>10.4f} "
                  f"{result['Precision']:>10.4f} "
                  f"{result['Recall']:>10.4f}")
        
        print("=" * 70)

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate YOLOv11 Fire Detection Model')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model weights')
    parser.add_argument('--test-images', action='store_true',
                       help='Test trên ảnh mẫu')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Số lượng ảnh test')
    parser.add_argument('--save', action='store_true',
                       help='Lưu kết quả')
    
    args = parser.parse_args()
    
    # Evaluate
    evaluate_model(model_path=args.model, save_results=args.save)
    
    # Test on images
    if args.test_images:
        output_dir = config.OUTPUT_DIR / 'evaluation'
        save_dir = output_dir if args.save else None
        test_on_images(
            model_path=args.model,
            num_samples=args.num_samples,
            save_dir=save_dir
        )

if __name__ == '__main__':
    main()
