import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO



model=YOLO('yolov8s.pt')


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap=cv2.VideoCapture(1)
# Set the resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Replace with your camera's supported width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Replace with your camera's supported height

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
print(class_list)
count=0
while True:
    
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1280,720))

    results=model.predict(frame)
    #print(results)
    a=results[0].boxes.data
    #print(a)

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
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, str(c), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)

    # Update the person_detected flag
    person_detected = new_person_detected

    # Display the frame
    cv2.imshow("Tello Feed", img)
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break


cap.release()
cv2.destroyAllWindows()
