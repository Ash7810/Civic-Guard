from ultralytics import YOLO
model = YOLO(r'C:\Users\Lenovo\runs\segment\Pothole_NewTraining_Optimized\weights\best.pt')

model.train(
    data= r'C:\Users\Lenovo\Desktop\CivicGuard\Pothole_Segmentation_YOLOv8\data.yaml',
    epochs=200,
    imgsz=640,
    batch=8,
    mosaic=1.0,
    mixup=0.2,
    name='Pothole_Detection'
)