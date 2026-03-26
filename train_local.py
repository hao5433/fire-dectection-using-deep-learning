"""
Huấn luyện model YOLOv11 cho phát hiện đám cháy
Train YOLOv11 Fire Detection Model
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO
import config
import utils

def train_model():
    """Huấn luyện YOLOv11 model"""
    print("\n" + "=" * 70)
    print("HUẤN LUYỆN MODEL PHÁT HIỆN ĐÁM CHÁY")
    print("=" * 70)
    
    # In cấu hình
    config.print_config()
    
    # Tạo thư mục output
    utils.create_output_dirs()
    
    # Kiểm tra data.yaml
    if not config.DATA_YAML.exists():
        print(f"❌ Không tìm thấy file: {config.DATA_YAML}")
        return
    
    print(f"\n✅ Dataset config: {config.DATA_YAML}")
    
    # Load model
    print(f"\n🔄 Đang load model {config.MODEL_NAME}...")
    model = YOLO(config.MODEL_NAME)
    print(f"✅ Đã load model thành công!")
    
    # Training configuration
    train_config = config.get_config()
    train_config.update({
        'project': str(config.RUNS_DIR),
        'name': 'fire_detection',
        'exist_ok': True,
        'save': True,
        'save_period': 10,  # Lưu checkpoint mỗi 10 epochs
    })
    
    print("\n📋 Training Configuration:")
    print("-" * 70)
    for key, value in train_config.items():
        print(f"  {key:20s}: {value}")
    print("-" * 70)
    
    # Bắt đầu training
    print("\n🚀 Bắt đầu training...")
    print("=" * 70)
    
    try:
        results = model.train(**train_config)
        
        print("\n" + "=" * 70)
        print("✅ TRAINING HOÀN TẤT!")
        print("=" * 70)
        
        # In kết quả
        results_dir = Path(config.RUNS_DIR) / 'fire_detection'
        print(f"\n📁 Kết quả lưu tại: {results_dir}")
        print(f"📁 Best model: {results_dir / 'weights' / 'best.pt'}")
        print(f"📁 Last model: {results_dir / 'weights' / 'last.pt'}")
        
        # In summary
        utils.print_training_summary(results_dir)
        
        return results
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training bị dừng bởi người dùng")
        return None
    except Exception as e:
        print(f"\n❌ Lỗi trong quá trình training: {e}")
        import traceback
        traceback.print_exc()
        return None

def resume_training(weights_path: str = None):
    """Tiếp tục training từ checkpoint"""
    if weights_path is None:
        # Tìm last.pt mới nhất
        runs_dir = config.RUNS_DIR / 'fire_detection'
        weights_path = runs_dir / 'weights' / 'last.pt'
    
    if not Path(weights_path).exists():
        print(f"❌ Không tìm thấy checkpoint: {weights_path}")
        return
    
    print(f"\n🔄 Tiếp tục training từ: {weights_path}")
    
    model = YOLO(str(weights_path))
    
    train_config = config.get_config()
    train_config.update({
        'project': str(config.RUNS_DIR),
        'name': 'fire_detection',
        'exist_ok': True,
        'resume': True,
    })
    
    try:
        results = model.train(**train_config)
        print("\n✅ Training hoàn tất!")
        return results
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return None

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train YOLOv11 Fire Detection Model')
    parser.add_argument('--resume', type=str, default=None, 
                       help='Path to checkpoint để tiếp tục training')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Số epochs (override config)')
    parser.add_argument('--batch', type=int, default=None,
                       help='Batch size (override config)')
    parser.add_argument('--imgsz', type=int, default=None,
                       help='Image size (override config)')
    parser.add_argument('--device', default=None,
                       help='Device: 0, 1, 2... hoặc cpu')
    
    args = parser.parse_args()
    
    # Override config nếu có arguments
    if args.epochs:
        config.EPOCHS = args.epochs
    if args.batch:
        config.BATCH_SIZE = args.batch
    if args.imgsz:
        config.IMAGE_SIZE = args.imgsz
    if args.device:
        config.DEVICE = args.device
    
    # Train hoặc resume
    if args.resume:
        resume_training(args.resume)
    else:
        train_model()

if __name__ == '__main__':
    main()
