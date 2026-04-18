import cv2
import numpy as np
import tensorflow as tf
import math
import winsound
import time
import os                          # for creating folders
import requests                    # for Telegram API calls
from datetime import datetime      # for timestamps
import folium                      # for map generation

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

# =========================
# Gender Prediction
# =========================
def predict_gender(face):

    if face is None or face.size == 0:
        return "Unknown"

    face = cv2.resize(face, (64,64))
    face = face / 255.0
    face = np.reshape(face,(1,64,64,3))

    pred = gender_model.predict(face,verbose=0)[0][0]

    return "Female" if pred < 0.55 else "Male"


# =========================
# Distance Calculation
# =========================
def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


# ==================================================
# NEW FEATURE CONFIG — edit these values before run
# ==================================================
os.makedirs("alerts", exist_ok=True)    # folder for screenshots

# --- Telegram ---
TELEGRAM_TOKEN   = "8069451506:AAHvF7eTb6wXRnpvwzWfoiQSkXHRg7rwjSI"   # from @BotFather
TELEGRAM_CHAT_ID = "1600203167"   # your personal chat ID

# --- Camera GPS (fixed location for this camera) ---
CAM_LAT = 28.4595
CAM_LON = 77.0266

# Keeps track of every alert point so the map shows all markers
alert_locations = []


# ==================================================
# NEW: Send Telegram message + screenshot
# ==================================================
def send_telegram_alert(image_path, male_count, female_count, timestamp):
    # Split timestamp (format: 20260415_143022) into readable date and time
    dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
    date_str = dt.strftime("%d %B %Y")          # e.g. 15 April 2026
    time_str = dt.strftime("%I:%M:%S %p")       # e.g. 02:30:22 PM
    maps_link = f"https://maps.google.com/?q={CAM_LAT},{CAM_LON}"

    # Formatted alert message
    message = (
        f"🔴 HIGH SECURITY ALERT 🔴\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⚠️  WOMAN IN DANGER  ⚠️\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"📅 Date       : {date_str}\n"
        f"🕐 Time       : {time_str}\n\n"
        f"👥 Males      : {male_count}\n"
        f"👩 Females    : {female_count}\n\n"
        f"📍 Location\n"
        f"   Lat : {CAM_LAT}\n"
        f"   Lon : {CAM_LON}\n"
        f"   🗺  {maps_link}\n\n"
        f"📸 Screenshot attached below."
    )

    # 1. Send the formatted text message
    requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
        data={"chat_id": TELEGRAM_CHAT_ID, "text": message},
        timeout=5
    )

    # 2. Send the screenshot with a short caption
    with open(image_path, "rb") as photo:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
            data={
                "chat_id": TELEGRAM_CHAT_ID,
                "caption": f"📸 Alert frame — {date_str}  {time_str}"
            },
            files={"photo": photo},
            timeout=10
        )


# =========================
# Start Webcam
# =========================
cap = cv2.VideoCapture(0)

ALERT_RADIUS = 200
last_beep_time = 0

while True:

    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    male_centers = []
    female_centers = []

    male_count = 0
    female_count = 0

    # =========================
    # YOLO Detection
    # =========================
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320, 320), swapRB=True)

    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for out in outs:
        for detection in out:

            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:

                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)

                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # =========================
    # Process detections safely
    # =========================
    if len(indexes) > 0:

        for i in indexes:

            if isinstance(i, (list, tuple, np.ndarray)):
                i = i[0]

            if i >= len(boxes):
                continue

            x, y, w, h = boxes[i]

            if classes[class_ids[i]] == "person":

                x1 = max(0, x)
                y1 = max(0, y)

                x2 = min(width, x+w)
                y2 = min(height, y+h)

                person = frame[y1:y2, x1:x2]

                gray = cv2.cvtColor(person, cv2.COLOR_BGR2GRAY)

                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.3,
                    minNeighbors=5,
                    minSize=(50,50)
                )

                gender = "Unknown"

                for (fx, fy, fw, fh) in faces:

                    face = person[fy:fy+fh, fx:fx+fw]

                    gender = predict_gender(face)

                    break

                cx = x + w//2
                cy = y + h//2

                if gender == "Male":
                    male_count += 1
                    male_centers.append((cx, cy))

                elif gender == "Female":
                    female_count += 1
                    female_centers.append((cx, cy))

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

                cv2.putText(
                    frame,
                    f"person ({gender})",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,255,0),
                    2
                )

    # =========================
    # ALERT LOGIC
    # =========================
    alert = False

    if len(female_centers) == 1:

        fx, fy = female_centers[0]

        nearby_men = sum(
            1 for mx, my in male_centers
            if distance((fx, fy), (mx, my)) < ALERT_RADIUS
        )

        if nearby_men >= 2:
            alert = True

    # =========================
    # Display Counts
    # =========================
    cv2.putText(frame, f"Males: {male_count}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,0,0),
                2)

    cv2.putText(frame, f"Females: {female_count}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,0,255),
                2)

    # =========================
    # SOUND ALERT
    # =========================
    if alert and time.time() - last_beep_time > 3:

        winsound.Beep(1200, 600)

        last_beep_time = time.time()

        # --------------------------------------------------
        # NEW FEATURE 1: Save screenshot with timestamp
        # --------------------------------------------------
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = f"alerts/alert_{timestamp}.jpg"
        cv2.imwrite(screenshot_path, frame)
        print(f"[+] Screenshot saved: {screenshot_path}")

        # --------------------------------------------------
        # NEW FEATURE 2: Send Telegram alert + screenshot
        # --------------------------------------------------
        try:
            send_telegram_alert(screenshot_path, male_count, female_count, timestamp)
            print("[+] Telegram alert sent")
        except Exception as e:
            print(f"[-] Telegram failed: {e}")

        # --------------------------------------------------
        # NEW FEATURE 3: Add marker on map, save as HTML
        # --------------------------------------------------
        alert_locations.append((CAM_LAT, CAM_LON))
        m = folium.Map(location=[CAM_LAT, CAM_LON], zoom_start=15)
        for lat, lon in alert_locations:
            folium.Marker(
                location=[lat, lon],
                popup=f"Alert at {timestamp}",
                icon=folium.Icon(color="red", icon="exclamation-sign")
            ).add_to(m)
        m.save("alert_map.html")
        print("[+] Map updated: alert_map.html")

    # =========================
    # ALERT DISPLAY
    # =========================
    if alert:

        cv2.rectangle(frame,
                      (0,0),
                      (width,height),
                      (0,0,255),
                      10)

        cv2.putText(frame,
                    "⚠ WOMAN SAFETY ALERT ⚠",
                    (80, height//2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0,0,255),
                    3)

    cv2.imshow("Women Safety Detection System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()