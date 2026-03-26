"""
Inference - Phát hiện lửa trên ảnh/video
Fire Detection Inference
"""

import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import config

class FireDetector:
    """Class để phát hiện lửa/khói"""
    
    def __init__(self, model_path: str = None, conf_threshold: float = None):
        """
        Khởi tạo Fire Detector
        
        Args:
            model_path: Đường dẫn đến model weights
            conf_threshold: Ngưỡng confidence
        """
        # Load model
        if model_path is None:
            model_path = config.RUNS_DIR / 'fire_detection' / 'weights' / 'best.pt'
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Không tìm thấy model: {model_path}")
        
        print(f"📦 Loading model từ: {model_path}")
        self.model = YOLO(str(model_path))
        
        self.conf_threshold = conf_threshold or config.CONFIDENCE_THRESHOLD
        self.iou_threshold = config.IOU_THRESHOLD
        
        print(f"✅ Model loaded successfully!")
        print(f"   Confidence threshold: {self.conf_threshold}")
        print(f"   IoU threshold: {self.iou_threshold}")
    
    def predict_image(self, image_path: str, save_path: str = None, show: bool = False):
        """
        Phát hiện lửa trên ảnh
        
        Args:
            image_path: Đường dẫn ảnh input
            save_path: Đường dẫn lưu kết quả (optional)
            show: Hiển thị kết quả
        
        Returns:
            results: YOLO results object
        """
        print(f"\n🔍 Detecting fire in: {image_path}")
        
        # Predict
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        result = results[0]
        boxes = result.boxes
        
        # Print detections
        print(f"   Found {len(boxes)} objects:")
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            # Xử lý trường hợp cls_id nằm ngoài phạm vi CLASS_NAMES
            if cls_id < len(config.CLASS_NAMES):
                cls_name = config.CLASS_NAMES[cls_id]
            else:
                cls_name = f"class_{cls_id}"
            print(f"     - {cls_name}: {conf:.2%}")
        
        # Visualize
        annotated = result.plot()
        
        if save_path:
            cv2.imwrite(save_path, annotated)
            print(f"   ✅ Saved to: {save_path}")
        
        if show:
            cv2.imshow('Fire Detection', annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return results
    
    def predict_video(self, video_path: str, output_path: str = None, 
                     show: bool = True, skip_frames: int = 0):
        """
        Phát hiện lửa trong video
        
        Args:
            video_path: Đường dẫn video input
            output_path: Đường dẫn lưu video kết quả
            show: Hiển thị kết quả real-time
            skip_frames: Bỏ qua N frames (tăng tốc)
        """
        print(f"\n🎥 Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Không thể mở video: {video_path}")
            return
        
        # Lấy thông tin video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"   Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        detection_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Skip frames nếu cần
                if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                    if writer:
                        writer.write(frame)
                    continue
                
                # Predict
                results = self.model.predict(
                    source=frame,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    verbose=False
                )
                
                # Visualize
                annotated = results[0].plot()
                
                # Count detections
                if len(results[0].boxes) > 0:
                    detection_count += 1
                
                # Write frame
                if writer:
                    writer.write(annotated)
                
                # Show
                if show:
                    cv2.imshow('Fire Detection', annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\n⚠️  Stopped by user")
                        break
                
                # Progress
                if frame_count % 30 == 0:
                    print(f"   Processed: {frame_count}/{total_frames} frames "
                          f"({frame_count/total_frames*100:.1f}%)")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
        
        print(f"\n✅ Video processing complete!")
        print(f"   Total frames: {frame_count}")
        print(f"   Frames with detections: {detection_count} ({detection_count/frame_count*100:.1f}%)")
        
        if output_path:
            print(f"   Output saved to: {output_path}")
    
    def predict_folder(self, folder_path: str, output_folder: str = None, 
                      extensions: list = None):
        """
        Phát hiện lửa trên tất cả ảnh trong folder
        
        Args:
            folder_path: Đường dẫn thư mục chứa ảnh
            output_folder: Thư mục lưu kết quả
            extensions: List các extension file ảnh
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        folder = Path(folder_path)
        
        if not folder.exists():
            print(f"❌ Không tìm thấy thư mục: {folder_path}")
            return
        
        # Tìm tất cả ảnh
        image_files = []
        for ext in extensions:
            image_files.extend(folder.glob(f'*{ext}'))
        
        if not image_files:
            print(f"❌ Không tìm thấy ảnh trong: {folder_path}")
            return
        
        print(f"\n📂 Processing {len(image_files)} images in: {folder_path}")
        
        # Tạo output folder
        if output_folder:
            output_path = Path(output_folder)
            output_path.mkdir(exist_ok=True, parents=True)
        
        # Process từng ảnh
        total_detections = 0
        
        for idx, img_path in enumerate(image_files, 1):
            print(f"\n[{idx}/{len(image_files)}] {img_path.name}")
            
            save_path = None
            if output_folder:
                save_path = output_path / img_path.name
            
            results = self.predict_image(str(img_path), save_path=str(save_path))
            total_detections += len(results[0].boxes)
        
        print(f"\n✅ Hoàn tất!")
        print(f"   Processed: {len(image_files)} images")
        print(f"   Total detections: {total_detections}")
        if output_folder:
            print(f"   Results saved to: {output_folder}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fire Detection Inference')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model weights')
    parser.add_argument('--source', type=str, required=True,
                       help='Path to image/video/folder')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save output')
    parser.add_argument('--conf', type=float, default=None,
                       help='Confidence threshold')
    parser.add_argument('--show', action='store_true',
                       help='Show results')
    parser.add_argument('--skip-frames', type=int, default=0,
                       help='Skip N frames (video only)')
    
    args = parser.parse_args()
    
    # Khởi tạo detector
    detector = FireDetector(model_path=args.model, conf_threshold=args.conf)
    
    source_path = Path(args.source)
    
    # Phát hiện loại source
    if source_path.is_file():
        # Check if video or image
        video_exts = ['.mp4', '.avi', '.mov', '.mkv']
        
        if source_path.suffix.lower() in video_exts:
            # Video
            detector.predict_video(
                video_path=args.source,
                output_path=args.output,
                show=args.show,
                skip_frames=args.skip_frames
            )
        else:
            # Image
            detector.predict_image(
                image_path=args.source,
                save_path=args.output,
                show=args.show
            )
    
    elif source_path.is_dir():
        # Folder
        detector.predict_folder(
            folder_path=args.source,
            output_folder=args.output
        )
    
    else:
        print(f"❌ Không tìm thấy: {args.source}")

if __name__ == '__main__':
    main()
