"""
REST API for Women Safety Detection System
GET /detect  - Run detection once from webcam, return JSON
POST /detect - Send image file, run detection on it, return JSON
"""

import cv2
import numpy as np
import tensorflow as tf
import math
import os
import requests                    # for ip-api geolocation call
from datetime import datetime
import time
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# =========================
# Initialize Flask
# =========================
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# =========================
# Load Models (same as main.py)
# =========================
print("[*] Loading YOLOv3-Tiny...")
net = cv2.dnn.readNet(
    "yolov3/yolov3-tiny.weights",
    "yolov3/yolov3-tiny.cfg"
)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

print("[*] Loading class names...")
classes = open("yolov3/coco.names").read().strip().split("\n")

print("[*] Loading gender model...")
gender_model = tf.keras.models.load_model("model/gender_model.h5")

print("[*] Loading face cascade...")
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# =========================
# Configuration (from main.py)
# =========================
ALERT_RADIUS = 200
GENDER_THRESHOLD = 0.65   # ← lower = more Female, raise toward 1.0 = more Male

# FIX 3: Location control — set USE_MANUAL_LOCATION = True to use exact coords below
#         set False to auto-fetch via IP geolocation
USE_MANUAL_LOCATION = True   # ← change this flag
CAM_LAT = 28.4595            # ← your exact latitude
CAM_LON = 77.0266            # ← your exact longitude

# =========================
# Helper Functions (same as main.py)
# =========================

# FIX 3: Location — manual override or IP-based fallback
def get_current_location():
    """
    Returns (lat, lon):
    - If USE_MANUAL_LOCATION is True  → uses CAM_LAT / CAM_LON exactly
    - If USE_MANUAL_LOCATION is False → fetches from ip-api.com, falls back to manual
    """
    if USE_MANUAL_LOCATION:
        print(f"[+] Using manual location: ({CAM_LAT}, {CAM_LON})")
        return CAM_LAT, CAM_LON

    # IP-based geolocation
    try:
        response = requests.get("http://ip-api.com/json/", timeout=5)
        data = response.json()

        if data.get("status") == "success":
            lat = data["lat"]
            lon = data["lon"]
            city = data.get("city", "Unknown")
            print(f"[+] Location fetched via IP: {city} ({lat}, {lon})")
            return lat, lon
        else:
            print(f"[-] Location API returned non-success: {data}")
    except Exception as e:
        print(f"[-] Location fetch failed: {e}")

    print("[!] Falling back to manual location")
    return CAM_LAT, CAM_LON


# FIX 1: Improved gender prediction with debug output
def predict_gender(face):
    """Predict gender from a face crop — identical to main.py"""
    if face is None or face.size == 0:
        print("[!] predict_gender: received empty face")
        return "Unknown"

    # FIX: Align with main.py — no extra type conversion, same ops in same order
    face = cv2.resize(face, (64, 64))          # resize to 64x64
    face = face / 255.0                        # normalize [0,1] — matches main.py exactly
    face = np.reshape(face, (1, 64, 64, 3))   # add batch dim

    pred = gender_model.predict(face, verbose=0)[0][0]
    print(f"[DEBUG] pred={pred:.4f}")

    return "Male" if pred < GENDER_THRESHOLD else "Female"


def distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def run_detection_once(frame):
    """
    Run detection logic ONCE on a single frame
    
    Args:
        frame: OpenCV image (BGR)
    
    Returns:
        dict: Detection results
    """
    if frame is None:
        return {
            "alert": False,
            "male_count": 0,
            "female_count": 0,
            "location": {"lat": CAM_LAT, "lon": CAM_LON},
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "error": "No frame captured"
        }
    
    height, width = frame.shape[:2]
    male_centers = []
    female_centers = []
    male_count = 0
    female_count = 0
    uncertain_preds = []  # FIX: collect uncertain predictions for fallback
    
    # FIX: Detect faces globally on full frame (not inside YOLO person boxes)
    # This correctly handles group photos, collages, and multi-person scenes

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(50, 50)
    )

    print(f"[+] Faces detected in frame: {len(faces)}")

    for (x, y, w, h) in faces:

        # Skip very small detections (noise)
        if w < 40 or h < 40:
            print(f"  [FACE] skipped (too small): size=({w},{h})")
            continue

        # Crop face directly from full BGR frame
        face_crop = frame[y:y+h, x:x+w]

        if face_crop.size == 0:
            continue

        # Same preprocessing as main.py
        face_input = cv2.resize(face_crop, (64, 64))
        face_input = face_input / 255.0
        face_input = np.reshape(face_input, (1, 64, 64, 3))

        pred = gender_model.predict(face_input, verbose=0)[0][0]

        # FIX: Stabilized gender logic — decision zones
        if pred < 0.48:
            gender = "Female"
        elif pred > 0.60:
            gender = "Male"
        else:
            gender = "Uncertain"

        print(f"  [DEBUG] pred={pred:.4f} → {gender}  face=({x},{y},{w},{h})")

        # Face center for proximity checks
        cx = x + w // 2
        cy = y + h // 2

        if gender == "Male":
            male_count += 1
            male_centers.append((cx, cy))
        elif gender == "Female":
            female_count += 1
            female_centers.append((cx, cy))
        else:
            # Uncertain — resolve with 0.55 midpoint
            if pred < 0.55:
                gender = "Female"
                female_count += 1
                female_centers.append((cx, cy))
            else:
                gender = "Male"
                male_count += 1
                male_centers.append((cx, cy))
            print(f"  [UNCERTAIN resolved] pred={pred:.4f} → {gender}")

    # ===== Check Alert Condition =====
    alert = False
    if len(female_centers) == 1:
        fx, fy = female_centers[0]
        nearby_men = sum(
            1 for mx, my in male_centers
            if distance((fx, fy), (mx, my)) < ALERT_RADIUS
        )
        if nearby_men >= 2:
            alert = True

    # FIX 2: Safety hazard logic (independent of alert)
    hazard = False
    hazard_message = ""
    if female_count >= 1 and male_count >= 2 * female_count:
        hazard = True
        hazard_message = "Safety hazard: multiple males surrounding female"
    
    # ===== If alert, save screenshot =====
    screenshot_path = None
    if alert:
        os.makedirs("alerts", exist_ok=True)
        timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = f"alerts/alert_{timestamp_file}.jpg"
        cv2.imwrite(screenshot_path, frame)
    
    # ===== Return Result =====
    # FIX 2: fetch live location instead of hardcoded coords
    lat, lon = get_current_location()

    return {
        "alert": alert,
        "hazard": hazard,
        "message": hazard_message,
        "male_count": male_count,
        "female_count": female_count,
        "location": {
            "lat": lat,
            "lon": lon,
            "maps_link": f"https://maps.google.com/?q={lat},{lon}"
        },
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "screenshot": screenshot_path if alert else None
    }


# =========================
# API Endpoints
# =========================

@app.route('/detect', methods=['GET'])
def detect_from_webcam():
    """
    GET /detect - Run detection from webcam
    
    Returns JSON with detection results
    """
    try:
        # ---- OLD CODE ----
        # cap = cv2.VideoCapture(0)
        # ret, frame = cap.read()
        # cap.release()

        # ---- NEW CODE: Capture frames for 10 seconds (Windows fix) ----

        # cv2.CAP_DSHOW = DirectShow backend — required on Windows
        # Without it, VideoCapture(0) silently fails to initialise
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not cap.isOpened():
            print("[-] ERROR: Could not open webcam")
            return jsonify({
                "error": "Failed to open webcam",
                "status": "error"
            }), 400

        print("[+] Camera opened successfully")

        # Warmup: give the camera 1 second to adjust exposure/white-balance
        # Without this, early frames are often black or blurry
        time.sleep(1)

        print("[*] Starting webcam capture...")

        frames = []               # store all valid frames
        start_time = time.time()  # mark start time
        CAPTURE_DURATION = 10     # seconds to capture
        failed_reads = 0          # track bad frames

        while time.time() - start_time < CAPTURE_DURATION:
            ret, frame = cap.read()

            if ret and frame is not None and frame.size > 0:
                # Extra checks: frame must be non-None and have actual pixel data
                frames.append(frame)

                # Live preview window (debug — closes after loop)
                cv2.imshow("Webcam Preview - Capturing...", frame)
                cv2.waitKey(1)
            else:
                failed_reads += 1
                print(f"[-] Frame read failed (total failures: {failed_reads})")

                # If camera keeps failing for 5 consecutive reads, break early
                if failed_reads >= 5 and len(frames) == 0:
                    print("[-] Camera unresponsive — stopping early")
                    break

            time.sleep(0.05)      # ~20 FPS, reduces CPU load

        cv2.destroyAllWindows()   # close preview window cleanly
        cap.release()             # always release camera

        elapsed = round(time.time() - start_time, 1)
        print(f"[+] Capture ended  | duration: {elapsed}s")
        print(f"[+] Frames captured: {len(frames)} good, {failed_reads} failed")

        # Safety check: must have at least 1 valid frame
        if not frames:
            return jsonify({
                "error": f"No valid frames captured ({failed_reads} failed reads)",
                "tip": "Check camera is not in use by another app",
                "status": "error"
            }), 400

        # Pick the best frame:
        # Skip early frames (camera still warming up) and use last 20% of frames
        # e.g. 200 frames → use frames[160:] then take the last one
        cutoff = max(0, int(len(frames) * 0.8))
        best_frame = frames[cutoff:][-1]   # last frame from the stable portion

        print(f"[+] Using frame {cutoff + len(frames[cutoff:])}/{len(frames)} as best frame")
        # ---- END NEW CODE ----

        # Run detection on the best frame
        result = run_detection_once(best_frame)
        result['status'] = 'success'
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500


@app.route('/detect', methods=['POST'])
def detect_from_image():
    """
    POST /detect - Upload image and run detection
    
    Expected: form-data with 'image' file
    Returns JSON with detection results
    """
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({
                "error": "No 'image' field in request",
                "status": "error"
            }), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({
                "error": "No file selected",
                "status": "error"
            }), 400
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read image
        frame = cv2.imread(filepath)
        
        if frame is None:
            return jsonify({
                "error": "Invalid image file",
                "status": "error"
            }), 400
        
        # Run detection
        result = run_detection_once(frame)
        result['status'] = 'success'
        result['uploaded_file'] = filename
        
        # Clean up
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "message": "Women Safety Detection API is running",
        "endpoints": {
            "GET": "/detect (from webcam)",
            "POST": "/detect (from uploaded image)",
            "GET": "/health (this endpoint)"
        }
    }), 200


# =========================
# Run Server
# =========================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Women Safety Detection API")
    print("="*60)
    print("[+] Server starting on http://localhost:5000")
    print("[+] GET  /detect     - Run detection from webcam")
    print("[+] POST /detect     - Run detection from uploaded image")
    print("[+] GET  /health     - Health check")
    print("="*60 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
