from ultralytics import YOLO
import os

def start_training():
    # Use small model for better accuracy vs speed balance
    model = YOLO('yolov8s-seg.pt')

    # Dataset path
    dataset_yaml = os.path.join(
        os.getcwd(),
        'Pothole_Segmentation_YOLOv8',
        'data.yaml'
    )

    model.train(
        # 📂 Data
        data=dataset_yaml,

        # 🧠 Training length
        epochs=250,
        patience=50,

        # 🖼 Image size (keep same as dataset)
        imgsz=640,

        # ⚙️ Hardware optimization
        batch=6,
        device=0,
        workers=4,
        cache='disk',      # IMPORTANT: prevents RAM crash

        # ⚡ Performance
        amp=True,

        # 🔥 Augmentation (key for potholes)
        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.3,
        fliplr=0.5,
        flipud=0.1,
        degrees=5.0,
        scale=0.5,
        shear=2.0,
        close_mosaic=0,    # keep mosaic ON till end

        # 🧩 Regularization
        dropout=0.0,
        weight_decay=0.0007,

        # 📈 Learning optimization
        cos_lr=True,
        warmup_epochs=5,
        lr0=0.01,
        lrf=0.01,

        # 🎯 Segmentation specific
        overlap_mask=True,

        # 📝 Logging
        name='Pothole_Final_640',
        exist_ok=True
    )

if __name__ == '__main__':
    start_training()