import cv2
import numpy as np
from ultralytics import YOLO  # Import the YOLO library

# Load the YOLOv8 model (make sure to download and specify the 'yolov8s-seg.pt' model)
model = YOLO('yolov8s-seg.pt')

def segment_humans_with_transparency(frame, alpha=0.5):
    # Run inference on the frame
    results = model(frame, task='segment')

    # Extract the masks and class IDs
    masks = results[0].masks.data.cpu().numpy() if results[0].masks is not None else []
    class_ids = results[0].boxes.cls.cpu().numpy() if results[0].boxes is not None else []

    # Create a blank canvas for human detection
    human_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

    for idx, class_id in enumerate(class_ids):
        if class_id == 0:  # Class ID 0 corresponds to "person" in COCO dataset
            human_mask = cv2.bitwise_or(human_mask, (masks[idx] * 255).astype(np.uint8))

    # Normalize the human mask to span the entire 0-255 range
    normalized_mask = cv2.normalize(human_mask, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Apply a sonar-like heatmap gradient using a colormap
    sonar_heatmap = cv2.applyColorMap(normalized_mask, cv2.COLORMAP_JET)

    # Overlay the heatmap on the original frame
    output = cv2.addWeighted(sonar_heatmap, 0.9, frame, 0.1, 0)

    return output


# Open the webcam (camera index 0 is usually the default camera)
cap = cv2.VideoCapture(1)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture video.")
        break

    # Resize frame for better performance (optional)
    frame = cv2.resize(frame, (640, 480))

    # Perform human segmentation with transparency
    output = segment_humans_with_transparency(frame, alpha=0.5)

    # Display the result
    cv2.imshow('Human Segmentation with Transparency', output)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
