import os
import cv2
import pygame
import pandas as pd
import numpy as np
from ultralytics import YOLO
from djitellopy import Tello
import keypressModule as kp
import cvzone
from tracker import Tracker

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

# Initialize Pygame and create a window
pygame.init()
window_width, window_height = 1280, 1080
window = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Tello Video Feed with Person Counting")

# Tracker initialization
tracker = Tracker()
cy1 = 120  # Line y-coordinate for counting
offset = 6
counter = set()  # Use set to avoid duplicate counts

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
    

    # Resize the frame for processing
    img_resized = cv2.resize(img, (320, 240))
    

    # Run YOLO model on the frame
    results = model.predict(img_resized)
    detections = results[0].boxes.data
    px = pd.DataFrame(detections).astype("float")

    # List for holding bounding boxes of "person" detections
    list_bbox = []

    # Loop through the detections
    for index, row in px.iterrows():
        x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        d = int(row[5])
        c = class_list[d]

        # If the class is "person", track and draw bounding box and text
        if c == "person":
            list_bbox.append([x1, y1, x2, y2])
            # Draw bounding box
            cv2.rectangle(img_resized, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # Draw text "person"
            cv2.putText(img_resized, str(c), (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)

    # Track and count persons crossing the line
    bbox_id = tracker.update(list_bbox)
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx, cy = int((x3 + x4) / 2), int((y3 + y4) / 2)

        if cy1 - offset < cy < cy1 + offset:
            if id not in counter:
                counter.add(id)  # Use set to store unique IDs

    # Convert to grayscale for the second display
    gray_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)  # Convert back to BGR for consistency

    # Frame for person counting
    counting_frame = img_resized.copy()
    cv2.line(counting_frame, (0, cy1), (320, cy1), (0, 255, 0), 2)
    cvzone.putTextRect(counting_frame, f'Count: {len(counter)}', (10, 30), scale=1, thickness=2)

    # Convert OpenCV frames to Pygame surfaces
    img_surface = cv2_to_pygame(img_resized)
    gray_img_surface = cv2_to_pygame(gray_img)
    counting_surface = cv2_to_pygame(counting_frame)

    # Clear the window
    window.fill((0, 0, 0))

    # Blit (draw) the color, grayscale, and counting frames onto the Pygame window
    window.blit(img_surface, (50, 50))           # Position the color frame
    window.blit(gray_img_surface, (400, 50))    # Position grayscale frame
    window.blit(counting_surface, (750, 50))    # Position counting frame
# Flip the entire window to mirror it horizontally
    flipped_window = pygame.transform.flip(window, True, False)  # Flip horizontally (True means horizontally)
    window.blit(flipped_window, (0, 0))  # Blit the flipped window back to the screen

    # Update the display
    pygame.display.update()

# Cleanup
tello.streamoff()
pygame.quit()
cv2.destroyAllWindows()
