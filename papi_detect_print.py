from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO("/home/vai/VAI2/ml_examples/labelImg/runs/detect/train5/weights/best.pt")

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Failed to open camera")
    exit()

# Class names (same order as your yaml)
class_names = ["low", "good", "high"]

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed")
        break

    # Run YOLO prediction on the frame
    results = model.predict(frame, conf=0.4, verbose=False)

    # Process results
    for r in results:
        boxes = r.boxes

        for box in boxes:
            cls = int(box.cls[0])
            label = class_names[cls]

            # Draw box
            x1, y1, x2, y2 = box.xyxy[0]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.putText(frame, label, (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            print("Detected:", label)

    # Show camera window
    cv2.imshow("PAPI Live Detection", frame)

    # Quit with Q key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
