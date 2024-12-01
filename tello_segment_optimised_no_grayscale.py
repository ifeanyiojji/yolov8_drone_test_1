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

# Load the YOLO model for object detection
model = YOLO('yolov8s.pt')

# Load the YOLO model for segmentation (to generate the heatmap)
segmentation_model = YOLO('yolov8s-seg.pt')

# Load class labels from coco.txt file
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Initialize Pygame and create a window
pygame.init()
window_width, window_height = 1280, 1080
window = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Tello Video Feed with Person Counting and Heatmap")

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

def segment_humans_with_transparency(frame, alpha=0.5):
    # Run inference on the frame
    results = segmentation_model(frame, task='segment')

    # Check if results have valid segmentation masks
    if results[0].masks is None or len(results[0].masks.data) == 0:
        print("No masks detected.")
        return frame  # Return the original frame if no masks are found

    # Extract the masks
    masks = results[0].masks.data.cpu().numpy()

    # Create a blank canvas for human detection
    human_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

    for mask in masks:
        resized_mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        human_mask = cv2.bitwise_or(human_mask, (resized_mask * 255).astype(np.uint8))

    # Normalize the human mask to span the entire 0-255 range
    normalized_mask = cv2.normalize(human_mask, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Apply a sonar-like heatmap gradient using a colormap
    sonar_heatmap = cv2.applyColorMap(normalized_mask, cv2.COLORMAP_JET)

    # Overlay the heatmap on the original frame with transparency
    output = cv2.addWeighted(sonar_heatmap, 0.9, frame, 0.1, 0)

    return output

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

    # Track and count persons crossing the line
    bbox_id = tracker.update(list_bbox)
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx, cy = int((x3 + x4) / 2), int((y3 + y4) / 2)

        if cy1 - offset < cy < cy1 + offset:
            if id not in counter:
                counter.add(id)  # Use set to store unique IDs

    # Frame for person counting
    counting_frame = img_resized.copy()
    cv2.line(counting_frame, (0, cy1), (320, cy1), (0, 255, 0), 2)
    cvzone.putTextRect(counting_frame, f'Count: {len(counter)}', (10, 30), scale=1, thickness=2)

    # Generate the heatmap frame
    heatmap_frame = segment_humans_with_transparency(img_resized)

    # Convert OpenCV frames to Pygame surfaces
    img_surface = cv2_to_pygame(img_resized)
    counting_surface = cv2_to_pygame(counting_frame)
    heatmap_surface = cv2_to_pygame(heatmap_frame)

    # Clear the window
    window.fill((0, 0, 0))

    # Blit (draw) the color, counting, and heatmap frames onto the Pygame window
    window.blit(img_surface, (50, 50))           # Position the color frame
    window.blit(counting_surface, (750, 50))    # Position counting frame
    window.blit(heatmap_surface, (50, 400))     # Position heatmap frame

    # Flip the entire window to mirror it horizontally
    flipped_window = pygame.transform.flip(window, True, False)  # Flip horizontally (True means horizontally)
    window.blit(flipped_window, (0, 0))  # Blit the flipped window back to the screen

    # Update the display
    pygame.display.update()

# Cleanup
tello.streamoff()
pygame.quit()
cv2.destroyAllWindows()
