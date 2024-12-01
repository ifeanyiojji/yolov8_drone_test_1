import os
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from djitellopy import Tello

# Connect to Tello
tello = Tello()
tello.connect()
tello.streamon()

# Load the YOLO model
model = YOLO('yolov8s.pt')

# Function to capture RGB values on mouse movement
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

# Set up the display window and mouse callback
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Load class labels from coco.txt file
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")
print(class_list)

# Initialize flag for person detection
person_detected = False
image_count = 0  # Counter for saved images

while True:
    # Get frame from Tello
    frame = tello.get_frame_read().frame
    if frame is None:
        break

    # Resize the frame
    frame = cv2.resize(frame, (720, 370))

    # Run YOLO model on the frame
    results = model.predict(frame)
    detections = results[0].boxes.data
    
    # Process detection results
    px = pd.DataFrame(detections).astype("float")
    
    new_person_detected = False  # Track if a person is found in this frame
    
    for index, row in px.iterrows():
        x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        d = int(row[5])
        c = class_list[d]
        
        if c == "person":
            new_person_detected = True
            
            # If a new person is detected, save the image in the "pictures" folder
            if not person_detected:
                image_count += 1
                image_filename = os.path.join("./pictures", f"person_detected_{image_count}.jpg")
                cv2.imwrite(image_filename, frame)
                print(f"New person detected, saving image as {image_filename}")

          
            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, str(c), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)

    # Update the person_detected flag
    person_detected = new_person_detected

    # Display the frame
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

# Release resources
tello.streamoff()
cv2.destroyAllWindows()
