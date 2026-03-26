"""
Script lấy metrics từ model YOLO đã train
Chạy: python get_metrics.py
"""

from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

def print_metrics_table(results):
    """In bảng metrics đẹp"""
    
    # Overall metrics
    map50 = results.box.map50
    map50_95 = results.box.map
    precision = results.box.mp
    recall = results.box.mr
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Interpretation
    def interpret(val):
        if val >= 0.8: return "Xuất sắc"
        elif val >= 0.7: return "Tốt"
        elif val >= 0.6: return "Khá"
        else: return "Cần cải thiện"
    
    print("\n" + "="*70)
    print("│" + " "*20 + "EVALUATION RESULTS" + " "*20 + "│")
    print("="*70)
    
    print("│   Overall Metrics:" + " "*49 + "│")
    print("│   ╔═══════════════╤════════════╤═══════════════════════════╗│")
    print("│   ║    Metric     │   Value    │   Interpretation          ║│")
    print("│   ╠═══════════════╪════════════╪═══════════════════════════╣│")
    print(f"│   ║ mAP50         │   {map50:.3f}    │   {map50*100:.1f}% - {interpret(map50):14s}║│")
    print(f"│   ║ mAP50-95      │   {map50_95:.3f}    │   {map50_95*100:.1f}% - {interpret(map50_95):14s}║│")
    print(f"│   ║ Precision     │   {precision:.3f}    │   {precision*100:.1f}% predict đúng      ║│")
    print(f"│   ║ Recall        │   {recall:.3f}    │   {recall*100:.1f}% phát hiện được    ║│")
    print(f"│   ║ F1-Score      │   {f1:.3f}    │   Cân bằng P/R            ║│")
    print("│   ╚═══════════════╧════════════╧═══════════════════════════╝│")
    
    # Per-class metrics
    print("│" + " "*68 + "│")
    print("│   Per-Class Performance:" + " "*43 + "│")
    print("│   ┌────────────┬────────┬────────┬────────┬─────────┐" + " "*11 + "│")
    print("│   │   Class    │ mAP50  │   P    │   R    │   F1    │" + " "*11 + "│")
    print("│   ├────────────┼────────┼────────┼────────┼─────────┤" + " "*11 + "│")
    
    names = results.names
    ap50_per_class = results.box.ap50
    p_per_class = results.box.p
    r_per_class = results.box.r
    ap_class_index = list(results.box.ap_class_index)
    
    
    best_f1_idx = -1
    best_f1 = 0
    
    # Calculate F1 for ALL classes in model
    class_data = []
    for cls_id in range(len(names)):
        cls_name = names[cls_id]
        
        # Check if this class has metrics
        if cls_id in ap_class_index:
            idx = ap_class_index.index(cls_id)
            ap50 = ap50_per_class[idx]
            p = p_per_class[idx]
            r = r_per_class[idx]
            f1_cls = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
        else:
            # Class không có trong kết quả (không detect được)
            ap50 = 0.0
            p = 0.0
            r = 0.0
            f1_cls = 0.0
            print(f"[WARNING] Class '{cls_name}' not detected in test set!")
        
        class_data.append((cls_name, ap50, p, r, f1_cls))
        if f1_cls > best_f1:
            best_f1 = f1_cls
            best_f1_idx = len(class_data) - 1
    
    for idx, (cls_name, ap50, p, r, f1_cls) in enumerate(class_data):
        star = " ⭐" if idx == best_f1_idx else "  "
        print(f"│   │ {cls_name:10s} │ {ap50:.3f}  │ {p:.3f}  │ {r:.3f}  │ {f1_cls:.3f}{star}│" + " "*11 + "│")
    
    print("│   └────────────┴────────┴────────┴────────┴─────────┘" + " "*11 + "│")
    print("="*70)
    
    # Return dict for further use
    return {
        'mAP50': map50,
        'mAP50-95': map50_95,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'per_class': class_data
    }

if __name__ == "__main__":
    import sys
    
    # Model path
    model_path = "runs/fire_detection/weights/best_kaggle.pt"
    
    # Check device
    device = 'cpu'
    try:
        import torch
        if torch.cuda.is_available():
            device = '0'
            print("🚀 Using GPU for evaluation")
        else:
            print("⚠️ GPU not available, using CPU (sẽ chậm ~10 phút)")
    except:
        pass
    
    print(f"\n📦 Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Verify class alignment between model and data.yaml
    import yaml
    with open('data.yaml') as f:
        cfg = yaml.safe_load(f)
    model_classes = list(model.names.values())
    yaml_classes = cfg['names']
    print(f"[CHECK] Model classes : {model_classes}")
    print(f"[CHECK] data.yaml     : {yaml_classes}")
    if model_classes != yaml_classes:
        print("[ERROR] CLASS MISMATCH! Fix data.yaml to match model before continuing.")
        print(f"        Expected: {model_classes}")
        import sys; sys.exit(1)
    else:
        print("[OK] Classes aligned correctly.")
    
    # Remove old cache files to avoid stale class mapping
    import glob, os
    for cache_file in glob.glob('**/*.cache', recursive=True):
        os.remove(cache_file)
        print(f"[CACHE] Deleted stale cache: {cache_file}")
    
    print(f"\n📊 Evaluating on test set...")
    print("   (Đợi vài phút nếu dùng CPU...)\n")
    
    # Run validation on full 4-class test split (no class filter — match training config)
    results = model.val(
        data='data.yaml',
        split='test',
        device=device,
        batch=8 if device == 'cpu' else 16,
        workers=0,
        verbose=False,
    )
    
    # Print formatted table
    metrics = print_metrics_table(results)
    
    print("\n✅ Evaluation complete!")
