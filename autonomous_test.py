import keyboard  # Make sure to install this library
from djitellopy import Tello
import time
import cv2

# Initialize Tello
me = Tello()
me.connect()
print("Battery:", me.get_battery(), "%")
me.streamon()

# Function to get coordinates from user
def get_coordinates():
    while True:
        try:
            start = input("Enter start coordinates (x y z) separated by space: ")
            start_coordinates = list(map(int, start.split()))
            if len(start_coordinates) != 3:
                print("Please enter exactly three integers for coordinates.")
                continue
            
            end = input("Enter end coordinates (x y z) separated by space: ")
            end_coordinates = list(map(int, end.split()))
            if len(end_coordinates) != 3:
                print("Please enter exactly three integers for coordinates.")
                continue
            
            return start_coordinates, end_coordinates
        except ValueError:
            print("Invalid input. Please enter valid integers.")

# Get coordinates from user
start_coordinates, end_coordinates = get_coordinates()
print("Start Coordinates set:", start_coordinates)
print("End Coordinates set:", end_coordinates)

# Control the drone manually
def get_keyboard_input():
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 50

    # Manual control commands
    if keyboard.is_pressed("left"): lr = -speed
    elif keyboard.is_pressed("right"): lr = speed

    if keyboard.is_pressed("up"): fb = speed
    elif keyboard.is_pressed("down"): fb = -speed

    if keyboard.is_pressed("w"): ud = speed
    elif keyboard.is_pressed("s"): ud = -speed

    if keyboard.is_pressed("a"): yv = speed
    elif keyboard.is_pressed("d"): yv = -speed

    if keyboard.is_pressed("q"): 
        me.land()
        time.sleep(1)

    if keyboard.is_pressed("t"):  # Press "t" to execute the coordinates
        execute_coordinates(start_coordinates, end_coordinates)

    return [lr, fb, ud, yv]

# Function to execute movement based on coordinates
def execute_coordinates(start, end):
    print("Taking off...")
    me.takeoff()
    time.sleep(5)  # Wait for the drone to stabilize

    print(f"Moving to End Coordinates: {end}")
    
    # Move in the x, y, and z directions based on the differences
    x_diff = end[0] - start[0]
    y_diff = end[1] - start[1]
    z_diff = end[2] - start[2]

    # Move to end coordinates
    me.send_rc_control(x_diff, y_diff, z_diff, 0)
    time.sleep(2)  # Wait for the drone to move

    print("Reached End Coordinates. Returning to Start Coordinates...")
    me.send_rc_control(-x_diff, -y_diff, -z_diff, 0)
    time.sleep(2)  # Wait for the drone to return

    print("Landing...")
    me.land()
    time.sleep(1)

# Main loop
while True:
    vals = get_keyboard_input()
    me.send_rc_control(vals[0], vals[1], vals[2], vals[3])
    
    img = me.get_frame_read().frame
    img = cv2.resize(img, (720, 560))
    cv2.imshow("image", img)
    cv2.waitKey(1)
