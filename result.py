from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO(r'C:\Users\Lenovo\runs\segment\Pothole_NewTraining_Optimized\weights\best.pt')

# Open video
cap = cv2.VideoCapture(r'C:\Users\Lenovo\Desktop\CivicGuard\Pothole_Segmentation_YOLOv8\sample_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)

    # Plot results (draw boxes + masks)
    annotated_frame = results[0].plot()

    # Count potholes
    if results[0].masks is not None:
        count = len(results[0].masks)
    else:
        count = 0

    # Show count on screen
    cv2.putText(
        annotated_frame,
        f'Potholes: {count}',
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    # Display video
    cv2.imshow("Pothole Detection", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()