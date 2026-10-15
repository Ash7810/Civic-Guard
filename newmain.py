from ultralytics import YOLO

def train_model():
    model = YOLO('yolov8s-seg.pt')

    model.train(
        data=r'C:\Users\Lenovo\Desktop\CivicGuard\Pothole_Segmentation_YOLOv8\data.yaml',
        
        # 🧠 Training
        epochs=200,
        patience=15,
        
        # 🖼 Image
        imgsz=640,
        
        # ⚙️ Hardware
        batch=6,
        device=0,
        workers=4,
        cache='disk',
        
        # ⚡ Learning
        optimizer='auto',
        lr0=0.01,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=5,
        
        # 🔥 Augmentation
        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.3,
        fliplr=0.5,
        flipud=0.1,
        degrees=5.0,
        scale=0.5,
        shear=2.0,
        close_mosaic=0,
        
        # 🧩 Regularization
        dropout=0.25,              # FIXED
        weight_decay=0.0007,
        
        # 🎯 Segmentation
        overlap_mask=True,
        
        # 📝 Misc
        seed=42,
        name='Pothole_NewTraining_Optimized',
        exist_ok=True
    )

if __name__ == '__main__':
    train_model()