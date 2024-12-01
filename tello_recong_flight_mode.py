import os
import cv2
import time
import numpy as np
import pygame
import pandas as pd
from ultralytics import YOLO
from djitellopy import Tello
import keypressModule as kp
import numpy as np

# Initialize keypress module
kp.init()

# Connect to Tello
tello = Tello()
tello.connect()
tello.streamon()
print(f"Battery Level: {tello.get_battery()}%")

# Load the YOLO model
model = YOLO('yolov8s.pt')

# Load class labels from coco.txt file
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Define the output folder path and create it if it doesn't exist
output_folder = "./pictures"
os.makedirs(output_folder, exist_ok=True)

# Initialize flags and counters
person_detected = False
image_count = 0  # Counter for saved images

# Initialize Pygame and create a window
pygame.init()
window_width, window_height = 1280, 960
window = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Tello Video Feed")

# Function to convert OpenCV frame to Pygame surface
def cv2_to_pygame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_surface = pygame.surfarray.make_surface(np.rot90(frame_rgb))
    return frame_surface

def getKeyboardInput():
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 50

    # Drone movement controls
    if kp.getKey("LEFT"): lr = -speed 
    elif kp.getKey("RIGHT"): lr = speed   
    if kp.getKey("UP"): fb = speed 
    elif kp.getKey("DOWN"): fb = -speed
    if kp.getKey("w"): ud = speed 
    elif kp.getKey("s"): ud = -speed
    if kp.getKey("a"): yv = speed 
    elif kp.getKey("d"): yv = -speed

    # Takeoff and landing
    if kp.getKey("t"): tello.takeoff()
    if kp.getKey("q"): tello.land()

    return [lr, fb, ud, yv]

running = True
while running:
    # Handle Pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Get keyboard input and control the Tello
    vals = getKeyboardInput()
    tello.send_rc_control(vals[0], vals[1], vals[2], vals[3])

    # Get frame from Tello
    img = tello.get_frame_read().frame
    if img is None:
        continue

    # Resize the frame for display
    img_resized = cv2.resize(img, (320, 240))

    # Run YOLO model on the frame
    results = model.predict(img_resized)
    detections = results[0].boxes.data

    # Process detection results for 'person' class only
    px = pd.DataFrame(detections).astype("float")
    new_person_detected = False

    for index, row in px.iterrows():
        x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        d = int(row[5])
        c = class_list[d]

        # Only process detections if the class is "person"
        if c == "person":
            new_person_detected = True

            # Save the frame if a new person is detected
            if not person_detected:
                image_count += 1
                image_filename = os.path.join(output_folder, f"person_detected_{image_count}.jpg")
                cv2.imwrite(image_filename, img)
                print(f"New person detected, saving image as {image_filename}")

            # Draw bounding box and label for persons only
            cv2.rectangle(img_resized, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img_resized, str(c), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)

    # Update the person_detected flag
    person_detected = new_person_detected

    # Convert to grayscale for the second display
    gray_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)  # Convert back to BGR for consistency

    # Convert OpenCV frames to Pygame surfaces
    img_surface = cv2_to_pygame(img_resized)
    gray_img_surface = cv2_to_pygame(gray_img)

    # Clear the window
    window.fill((0, 0, 0))

    # Blit (draw) the color and grayscale frames onto the Pygame window
    window.blit(img_surface, (50, 50))  # Position the color frame at (50, 50)
    window.blit(gray_img_surface, (400, 50))  # Position grayscale frame at (400, 50)
   # window.blit(img_surface, (50, 400))  # Position the color frame at (50, 50)
   # window.blit(gray_img_surface, (400, 400))  # Position grayscale frame at (400, 50)


    # Update the display
    pygame.display.update()

# Cleanup
tello.streamoff()
pygame.quit()
cv2.destroyAllWindows()
