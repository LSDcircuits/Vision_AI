# some ai made code which works somewhart

from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque

# -----------------------------
# LOAD YOLO MODEL
# -----------------------------
model = YOLO("/home/vai/VAI2/ml_examples/labelImg/runs/detect/train/weights/best.pt")

# CAMERA STREAM
stream_url = "http://vai:vai@192.168.178.37:8081/video"
cap = cv2.VideoCapture(stream_url)

# Smoothing buffer for classification stability
history = deque(maxlen=7)


# -----------------------------
#   WINDOW + SLIDER SETUP
# -----------------------------
def nothing(x):
    pass

cv2.namedWindow("PAPI Detector")
cv2.createTrackbar("Confidence", "PAPI Detector", 10, 50, nothing)
cv2.createTrackbar("Debug", "PAPI Detector", 0, 1, nothing)
# Debug = 0 → OFF, Debug = 1 → ON


# -----------------------------
#   CLASSIFY PAPI INSIDE CROP
# -----------------------------
def classify_papi(crop):

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # -------------------------
    # RED MASK
    # -------------------------
    red1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
    red_mask = cv2.bitwise_or(red1, red2)

    # -------------------------
    # WHITE MASK (Use brightness)
    # -------------------------
    white_mask = cv2.inRange(hsv, (0, 0, 210), (180, 40, 255))

    # Clean up noise
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)

    # -------------------------
    # READ SLIDER VALUES
    # -------------------------
    confidence = cv2.getTrackbarPos("Confidence", "PAPI Detector")
    debug_mode = cv2.getTrackbarPos("Debug", "PAPI Detector")

    min_blob_size = confidence            # dynamic threshold
    max_blob_size = confidence * 150      # dynamic scaling

    # -------------------------
    # COUNT BLOBS FUNCTION
    # -------------------------
    def count_blobs(mask, color):

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        count = 0

        for i, s in enumerate(stats[1:], start=1):
            area = s[cv2.CC_STAT_AREA]

            if min_blob_size < area < max_blob_size:
                count += 1

                if debug_mode == 1:
                    x, y, w, h, _ = s
                    cv2.rectangle(crop, (x, y), (x+w, y+h),
                                  (0, 0, 255) if color == "red" else (255, 255, 255), 1)

        return count

    # Count red/white LEDs
    red_count = count_blobs(red_mask, "red")
    white_count = count_blobs(white_mask, "white")

    # -------------------------
    # SHOW DEBUG WINDOWS
    # -------------------------
    if debug_mode == 1:
        cv2.imshow("RED MASK", red_mask)
        cv2.imshow("WHITE MASK", white_mask)
        cv2.imshow("CROP DEBUG", crop)

    # -------------------------
    # CLASSIFY USING PAPI RULES
    # -------------------------
    if red_count + white_count == 0:
        label = "UNKNOWN"
    else:
        if white_count == 4:
            label = "TOO HIGH"
        elif white_count == 3:
            label = "HIGH"
        elif white_count == 2:
            label = "GOOD"
        elif white_count == 1:
            label = "LOW"
        else:
            label = "TOO LOW"

    # Smooth output
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

            pad = 10
            crop = frame[max(0, y1-pad):y2+pad, max(0, x1-pad):x2+pad]
            if crop.size == 0:
                continue

            status = classify_papi(crop)

            # Draw bounding box + result on main image
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, status, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)

    cv2.imshow("PAPI Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
