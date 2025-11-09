from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque

# LOAD MODEL
model = YOLO("/home/vai/VAI2/ml_examples/labelImg/runs/detect/train/weights/best.pt")

# PHONE CAMERA STREAM WITH LOGIN
stream_url = "http://vai:vai@192.168.178.37:8081/video"
cap = cv2.VideoCapture(stream_url)

# Smoothing buffer
history = deque(maxlen=7)

def classify_papi(crop):

    crop = cv2.GaussianBlur(crop, (3,3), 0)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # RED
    red1 = cv2.inRange(hsv, (0, 90, 80), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 90, 80), (180, 255, 255))
    red_mask = cv2.bitwise_or(red1, red2)

    # WHITE
    white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 60, 255))

    # BRIGHT FILTER
    v = hsv[:, :, 2]
    bright = cv2.inRange(v, 180, 255)
    red_mask = cv2.bitwise_and(red_mask, bright)
    white_mask = cv2.bitwise_and(white_mask, bright)

    red = cv2.countNonZero(red_mask)
    white = cv2.countNonZero(white_mask)
    total = red + white

    ratio = red / total if total > 0 else 0.5

    # CLASSIFY
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

    # SMOOTH OUTPUT
    history.append(label)
    return max(set(history), key=history.count)


while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera disconnected!")
        break

    results = model(frame, stream=True)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Add padding
            pad = 10
            crop = frame[max(0, y1-pad):y2+pad, max(0, x1-pad):x2+pad]

            # Check for empty crop
            if crop.size == 0:
                continue

            status = classify_papi(crop)

            # DRAW
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, status, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("PAPI Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
