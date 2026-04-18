"""
Configuration file for Multi-Camera Safety Detection System
"""
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict

# ==========================================
# CAMERA CONFIGURATION
# ==========================================
@dataclass
class CameraConfig:
    """Camera configuration"""
    camera_id: str
    source: str  # 0 for webcam, path for video file, URL for IP camera
    latitude: float
    longitude: float
    name: str
    enabled: bool = True

# Define your cameras here
CAMERAS: List[CameraConfig] = [
    CameraConfig(
        camera_id="CAM_01",
        source=0,  # Webcam 0
        latitude=40.7128,   # NYC example
        longitude=-74.0060,
        name="Main Entrance"
    ),
    CameraConfig(
        camera_id="CAM_02",
        source="http://192.168.1.100:8080/video",  # IP camera example
        latitude=40.7130,
        longitude=-74.0065,
        name="Hallway"
    ),
    # Add more cameras as needed
]

# ==========================================
# ALERT RULES
# ==========================================
ALERT_RULES = {
    "min_females": 1,           # Alert if this many females detected
    "min_males": 2,             # Alert if this many males nearby
    "alert_radius": 200,        # Pixel distance threshold
    "min_confidence": 0.5,      # YOLO detection confidence
    "gender_threshold": 0.55,   # Gender model threshold
}

# ==========================================
# VIDEO BUFFER SETTINGS
# ==========================================
VIDEO_BUFFER_SECONDS = 30      # Keep last 30 seconds
FPS = 30                        # Frames per second (adjust based on camera)
BUFFER_SIZE = VIDEO_BUFFER_SECONDS * FPS

# ==========================================
# ALERT SETTINGS
# ==========================================
ALERT_OUTPUT_DIR = "alerts"    # Directory to save snapshots/videos
SNAPSHOT_DIR = os.path.join(ALERT_OUTPUT_DIR, "snapshots")
VIDEO_DIR = os.path.join(ALERT_OUTPUT_DIR, "videos")
LOG_DIR = os.path.join(ALERT_OUTPUT_DIR, "logs")

# Create directories if they don't exist
for directory in [ALERT_OUTPUT_DIR, SNAPSHOT_DIR, VIDEO_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

# ==========================================
# TELEGRAM ALERT SETTINGS
# ==========================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID_HERE")

# ==========================================
# EMAIL ALERT SETTINGS
# ==========================================
EMAIL_ENABLED = False
EMAIL_SENDER = "your_email@gmail.com"
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
EMAIL_RECIPIENT = "recipient@example.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# ==========================================
# HEATMAP SETTINGS
# ==========================================
HEATMAP_OUTPUT = "heatmap.html"
HEATMAP_CENTER_LAT = 40.7128
HEATMAP_CENTER_LON = -74.0060
HEATMAP_ZOOM = 14

# ==========================================
# MODEL PATHS
# ==========================================
YOLO_WEIGHTS = "yolov3/yolov3-tiny.weights"
YOLO_CFG = "yolov3/yolov3-tiny.cfg"
YOLO_NAMES = "yolov3/coco.names"
GENDER_MODEL_PATH = "model/gender_model.h5"

# ==========================================
# PROCESSING SETTINGS
# ==========================================
FRAME_RESIZE = (640, 480)      # Resize frame for faster processing
SKIP_FRAMES = 2                # Process every 2nd frame (improves speed)

# ==========================================
# PERFORMANCE SETTINGS
# ==========================================
NUM_THREADS = 4                # Number of threads for camera processing
ENABLE_GPU = False             # Use GPU if available

# ==========================================
# DASHBOARD SETTINGS
# ==========================================
DASHBOARD_PORT = 8501
FLASK_PORT = 5000
FLASK_HOST = "0.0.0.0"
