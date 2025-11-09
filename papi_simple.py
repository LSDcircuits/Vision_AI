# code works, simplest opencv + yolo script //better for understanding
# open YOLO model detects the papi, in the frame openVC counts pixels RED & WHITE labels LOW GOOD or HIGH depedning on ratio... only high & low work
from ultralytics import YOLO
import cv2
import numpy as np


model = YOLO("/home/vai/VAI2/ml_examples/labelImg/runs/detect/train/weights/best.pt")


stream_url = "http://vai:vai@192.168.178.37:8081/video"
cap = cv2.VideoCapture(stream_url)

from collections import deque
import numpy as np
import cv2

history = deque(maxlen=7)

def classify_papi(crop):

    crop = cv2.GaussianBlur(crop, (3,3), 0)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    red1 = cv2.inRange(hsv, (0, 90, 80), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 90, 80), (180, 255, 255))
    red_mask = cv2.bitwise_or(red1, red2)


    white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 60, 255))

   
    v = hsv[:, :, 2]
    bright = cv2.inRange(v, 180, 255)
    red_mask = cv2.bitwise_and(red_mask, bright)
    white_mask = cv2.bitwise_and(white_mask, bright)

    red = cv2.countNonZero(red_mask)
    white = cv2.countNonZero(white_mask)
    total = red + white

    if total == 0:
        ratio = 0.5  # neutral backup
    else:
        ratio = red / total

    
    if ratio < 0.25:
        label = "TOO HIGH"
    elif ratio < 0.40:
        label = "HIGH"
    elif ratio < 0.60:
        label = "GOOD"
    elif ratio < 0.75:
        label = "LOW"
    else:
        label = "TOO LOW"

  
    history.append(label)
    final_label = max(set(history), key=history.count)

    return final_label


while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera disconnected!")
        break

    results = model(frame, stream=True)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

        
            crop = frame[y1:y2, x1:x2]

            status = classify_papi(crop)

          
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, status, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("PAPI Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()
