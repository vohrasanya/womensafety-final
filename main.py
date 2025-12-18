import cv2
import numpy as np
import tensorflow as tf
import math
import winsound
import time

# =========================
# Load YOLOv3-Tiny
# =========================
net = cv2.dnn.readNet(
    "yolov3/yolov3-tiny.weights",
    "yolov3/yolov3-tiny.cfg"
)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
classes = open("yolov3/coco.names").read().strip().split("\n")

# =========================
# Load Gender Model
# =========================
gender_model = tf.keras.models.load_model("model/gender_model.h5")

# =========================
# Face Detector
# =========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def predict_gender(face):
    if face is None or face.size == 0:
        return "Unknown"
    h, w, _ = face.shape
    if h < 20 or w < 20:
        return "Unknown"

    face = cv2.resize(face, (64, 64))
    face = face / 255.0
    face = np.reshape(face, (1, 64, 64, 3))
    pred = gender_model.predict(face, verbose=0)[0][0]
    return "Male" if pred < 0.5 else "Female"

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# =========================
# Webcam
# =========================
cap = cv2.VideoCapture(0)

ALERT_RADIUS = 200
last_beep_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    male_centers, female_centers = [], []
    male_count, female_count = 0, 0

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320, 320), swapRB=True)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    for out in outs:
        for d in out:
            scores = d[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.5:
                cx, cy = int(d[0]*width), int(d[1]*height)
                w, h = int(d[2]*width), int(d[3]*height)
                x, y = int(cx-w/2), int(cy-h/2)
                boxes.append([x, y, w, h])
                confidences.append(float(conf))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in indexes:
        x, y, w, h = boxes[i]
        if classes[class_ids[i]] == "person":
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(width, x+w), min(height, y+h)
            person = frame[y1:y2, x1:x2]

            gray = cv2.cvtColor(person, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            gender = "Unknown"
            for (fx, fy, fw, fh) in faces:
                gender = predict_gender(person[fy:fy+fh, fx:fx+fw])
                break

            cx, cy = x+w//2, y+h//2

            if gender == "Male":
                male_count += 1
                male_centers.append((cx, cy))
            elif gender == "Female":
                female_count += 1
                female_centers.append((cx, cy))

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f"person ({gender})",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # =========================
    # ALERT LOGIC
    # =========================
    alert = False
    if len(female_centers) == 1:
        fx, fy = female_centers[0]
        nearby_men = sum(
            1 for mx,my in male_centers
            if distance((fx,fy),(mx,my)) < ALERT_RADIUS
        )
        if nearby_men >= 2:
            alert = True

    # =========================
    # DISPLAY COUNTS
    # =========================
    cv2.putText(frame, f"Males: {male_count}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    cv2.putText(frame, f"Females: {female_count}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # =========================
    # SOUND + ALERT BANNER
    # =========================
    if alert and time.time() - last_beep_time > 3:
        winsound.Beep(1200, 600)
        last_beep_time = time.time()

    if alert:
        cv2.rectangle(frame, (0,0), (width,height), (0,0,255), 10)
        cv2.putText(frame, "⚠ WOMAN SAFETY ALERT ⚠",
                    (80, height//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0,0,255), 3)

    cv2.imshow("Women Safety Detection System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
