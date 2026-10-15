import cv2
from ultralytics import YOLO

# 1. Load your trained model
model = YOLO(r'C:\Users\Lenovo\runs\segment\Pothole_Final_640\weights\best.pt')  # Update path if needed

# 2. Open the video file
video_path = r'C:\Users\Lenovo\Desktop\CivicGuard\Pothole_Segmentation_YOLOv8\sample_video.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()

    if success:
        # Run YOLO inference on the frame
        # stream=True is more memory-efficient for videos
        results = model(frame, conf=0.15, device=0)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("CivicGuard Live Detection", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()