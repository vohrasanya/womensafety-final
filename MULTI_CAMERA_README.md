# 🛡️ Multi-Camera Intelligent Safety Detection System

**Educational Computer Vision Project** | Women Safety Detection using YOLOv3 + Gender Classification + Multi-Camera Support

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [System Architecture](#system-architecture)
4. [Requirements](#requirements)
5. [Installation](#installation)
6. [Configuration](#configuration)
7. [Usage](#usage)
8. [Dashboard](#dashboard)
9. [File Structure](#file-structure)
10. [Troubleshooting](#troubleshooting)
11. [Performance Optimization](#performance-optimization)
12. [Security Notes](#security-notes)

---

## 📌 Overview

This is an **educational AI safety system** (not surveillance) that:
- Detects dangerous situations in real-time
- Monitors multiple camera feeds simultaneously
- Sends immediate alerts to designated recipients
- Maintains incident logs with GPS mapping
- Provides real-time dashboard visualization

### Educational Purpose
This system is designed for learning computer vision, alert systems, and distributed processing. It demonstrates how AI can be used for safety applications while respecting privacy.

---

## ✨ Features

### 1. **Multi-Camera Support**
- Handle 5-10+ concurrent CCTV cameras
- Each camera with unique ID and GPS coordinates
- Independent streams processed in parallel
- Non-blocking thread-based architecture

### 2. **Real-Time Detection**
- YOLOv3-Tiny for fast person detection
- Custom TensorFlow CNN for gender classification
- ~30+ FPS processing (varies by hardware)

### 3. **Location Mapping**
- GPS coordinates for all cameras
- Heatmap visualization of incidents
- Hotspot clustering
- Interactive maps using Folium

### 4. **Incident Recording**
- 30-second circular video buffer (auto-retains last 30 sec)
- Snapshots on alert trigger
- Automatic video clip extraction

### 5. **Multi-Channel Alerts**
- **Telegram Bot** - Instant notifications with image/video
- **Email** - Detailed report with attachments
- Alert logging and history
- Cooldown system to prevent spam

### 6. **Real-Time Dashboard**
- Live feed monitoring
- Alert statistics
- Incident map with heatmaps
- Snapshot gallery
- Alert log viewer
- CSV export

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────┐
│      Multi-Camera Safety Detection System           │
└──────────────────┬──────────────────────────────────┘
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
    CAM_01     CAM_02     CAM_03
    (GPS)      (GPS)      (GPS)
        │          │          │
        └──────────┼──────────┘
                   ▼
        ┌──────────────────────┐
        │ Multi-Camera Manager │   (camera_handler.py)
        │ - Stream handling    │
        │ - Thread management  │
        └──────────┬───────────┘
                   ▼
        ┌──────────────────────┐
        │  Alert Manager       │   (alert_manager.py)
        │ - Detection pipeline │
        │ - Rule engine        │
        │ - Video buffering    │
        └──────┬───────┬───────┘
               │       │
        ┌──────▼──┐   ▼─────────┐
        │ Snapshot│  Video Clip │
        │ Saver   │  Saver      │
        └─────┬───┘   ┴─────┬───┘
              │             │
        ┌─────▼─────────────▼───┐
        │  Alert Sender         │  (alert_sender.py)
        │ - Telegram Bot        │
        │ - Email Sender        │
        │ - Alert Logger        │
        └─────┬─────────┬───────┘
              │         │
        ┌─────▼──┐  ┌───▼──────┐
        │Telegram│  │Email     │
        │Bot API │  │SMTP      │
        └────────┘  └──────────┘

        ┌─────────────────────────┐
        │ Location Mapper         │  (location_mapper.py)
        │ - GPS plotting          │
        │ - Heatmap generation    │
        │ - Hotspot clustering    │
        └─────────────────────────┘

        ┌─────────────────────────┐
        │ Dashboard               │  (dashboard.py)
        │ - Streamlit UI          │
        │ - Real-time updates     │
        │ - Report generation     │
        └─────────────────────────┘
```

---

## 📦 Requirements

### Hardware
- **Minimum**: Desktop/Laptop with 4GB RAM
- **Recommended**: 8GB+ RAM, GPU (CUDA-capable for faster processing)
- **Network**: Sufficient bandwidth for multiple camera streams

### Software
- Python 3.8+
- OpenCV with DNN support
- TensorFlow 2.10+
- Streamlit (for dashboard)

### Models
- YOLOv3-Tiny weights (~33 MB)
- Gender classification model (gender_model.h5)

### API Keys
- Telegram Bot Token (free from @BotFather)
- Optional: Email account for alerts

---

## 🚀 Installation

### 1. Clone or Extract Project
```bash
cd e:\project\gender_1
```

### 2. Create Virtual Environment
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
# or: source venv/bin/activate  # Linux/Mac
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify YOLO Models Exist
```
Ensure these files exist:
- yolov3/yolov3-tiny.weights
- yolov3/yolov3-tiny.cfg
- yolov3/coco.names
- model/gender_model.h5

If missing, download from:
https://pjreddie.com/darknet/yolo/
```

### 5. Setup Environment Variables
```bash
# Copy the template
copy .env.example .env

# Edit .env with your settings:
TELEGRAM_BOT_TOKEN=your_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

---

## ⚙️ Configuration

### Configure Cameras (`config.py`)

```python
CAMERAS: List[CameraConfig] = [
    CameraConfig(
        camera_id="CAM_01",
        source=0,  # Webcam 0 or IP camera URL
        latitude=40.7128,
        longitude=-74.0060,
        name="Main Entrance"
    ),
    CameraConfig(
        camera_id="CAM_02",
        source="http://192.168.1.100:8080/video",
        latitude=40.7130,
        longitude=-74.0065,
        name="Hallway"
    ),
]
```

### Configure Alert Rules (`config.py`)

```python
ALERT_RULES = {
    "min_females": 1,           # Alert if 1+ females detected
    "min_males": 2,             # Alert if 2+ males nearby
    "alert_radius": 200,        # Pixel distance threshold
    "min_confidence": 0.5,      # YOLO detection confidence
    "gender_threshold": 0.55,   # Gender model threshold (0=female, 1=male)
}
```

### Configure Telegram Bot

**Step 1: Create Bot**
1. Message @BotFather on Telegram
2. Run `/newbot`
3. Follow prompts, get token

**Step 2: Get Chat ID**
Send a message to your bot, then visit:
```
https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates
```
Find `"id"` in the response

**Step 3: Update .env**
```
TELEGRAM_BOT_TOKEN=123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11
TELEGRAM_CHAT_ID=987654321
```

### Configure Email Alerts (Optional)

**For Gmail:**
1. Enable 2-Factor Authentication
2. Create App Password: https://myaccount.google.com/apppasswords
3. Update `config.py`:

```python
EMAIL_ENABLED = True
EMAIL_SENDER = "your_email@gmail.com"
EMAIL_RECIPIENT = "alert@example.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
```

4. Update `.env`:
```
EMAIL_PASSWORD=your_app_password_here
```

---

## 🎬 Usage

### Start Multi-Camera System

```bash
python multi_camera_main.py
```

**Output:**
```
🚀 Initializing Multi-Camera Safety Detection System...
✅ System initialized successfully!
📷 Setting up 2 camera(s)...
✅ Camera CAM_01 added to manager
✅ Camera CAM_02 added to manager
▶️ Starting system...
✅ Camera CAM_01 started
✅ Camera CAM_02 started
✅ System started! Processing feeds...
🎬 Processing thread started for CAM_01
🎬 Processing thread started for CAM_02
```

### View Real-Time Dashboard

```bash
streamlit run dashboard.py
```

Visit: `http://localhost:8501`

### Stop System
Press `Ctrl+C` in the main terminal

---

## 📊 Dashboard Features

### Views Available:
1. **Dashboard** - Statistics and recent activity
2. **Maps & Heatmaps** - Incident locations visualization
3. **Snapshots** - Latest alert images
4. **Alert Logs** - Detailed alert history with CSV export

### Features:
- 🔄 Real-time refresh
- 📥 Download reports as CSV
- 📍 Interactive maps
- 🔥 Heatmap visualization
- 📸 Snapshot gallery

---

## 📁 File Structure

```
project_root/
├── config.py                 # Configuration & camera setup
├── camera_handler.py         # Multi-camera stream management
├── detection_refactored.py   # YOLO + Gender classification
├── video_buffer.py           # Circular video buffer (30 sec)
├── alert_manager.py          # Alert orchestration
├── alert_sender.py           # Telegram & Email alerts
├── location_mapper.py        # GPS mapping & heatmaps
├── multi_camera_main.py      # Main orchestrator (START HERE)
├── dashboard.py              # Streamlit web dashboard
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variables template
├── .env                      # Your actual secrets (DONT COMMIT)
│
├── model/
│   ├── gender_model.h5       # Gender classification model
│   └── train_gender_model.py # Model training script
│
├── yolov3/
│   ├── yolov3-tiny.weights   # YOLO weights
│   ├── yolov3-tiny.cfg       # YOLO config
│   └── coco.names            # Class names
│
├── alerts/                   # Generated at runtime
│   ├── snapshots/            # Alert snapshots
│   ├── videos/               # Alert video clips
│   ├── alerts.log            # Alert log file
│   └── alerts.json           # Structured alert data
│
├── heatmap.html              # Generated heatmap
├── detailed_map.html         # Generated detailed map
└── alerts_data.json          # Exported alert data
```

---

## 🔧 Troubleshooting

### Issue: "Camera failed to open"
```
Solution:
1. Check camera_id value in config (0 for webcam, URL for IP camera)
2. Verify camera is not in use by other application
3. Test with: cv2.VideoCapture(0) in Python shell
```

### Issue: "Model not found"
```
Solution:
1. Verify gender_model.h5 exists: model/gender_model.h5
2. Download YOLO files:
   - yolov3-tiny.weights: https://pjreddie.com/darknet/yolo/
   - Place in: yolov3/ folder
```

### Issue: "Telegram alerts not sending"
```
Solution:
1. Verify bot token is correct in .env
2. Verify chat ID is correct
3. Test manually:
   curl -X POST https://api.telegram.org/botYOUR_TOKEN/sendMessage \
     -d chat_id=YOUR_CHAT_ID \
     -d text="Test"
```

### Issue: Low FPS / Slow Processing
```
Solution:
1. Reduce FRAME_RESIZE in config.py: (320, 240) instead of (640, 480)
2. Increase SKIP_FRAMES: Process every 3rd frame instead of 2nd
3. Use GPU if available:
   - Install CUDA: https://developer.nvidia.com/cuda-11.8-download
   - Install cuDNN
   - Update config.py: ENABLE_GPU = True
4. Reduce number of cameras temporarily
```

### Issue: High Memory Usage
```
Solution:
1. Reduce VIDEO_BUFFER_SECONDS in config.py (15 instead of 30)
2. Reduce FPS in config.py
3. Disable video-saving for all alerts
```

---

## ⚡ Performance Optimization

### For Real-Time Processing:

1. **Hardware Acceleration**
   ```python
   # config.py
   ENABLE_GPU = True
   YOLO_MODEL = "tiny"  # Use tiny model instead of full
   ```

2. **Frame Skipping**
   ```python
   SKIP_FRAMES = 3  # Process every 3rd frame
   FPS = 15  # Reduce to 15 FPS instead of 30
   ```

3. **Parallel Processing**
   ```python
   NUM_THREADS = 8  # More threads for more cameras
   ```

4. **Memory Management**
   ```python
   VIDEO_BUFFER_SECONDS = 15  # Shorter buffer
   FRAME_RESIZE = (320, 240)  # Smaller resolution
   ```

### Benchmarks (Single Camera):
- YOLOv3-Tiny on CPU: ~20-30 FPS
- With GPU (CUDA): ~100+ FPS
- Per-camera memory: ~150-300 MB

---

## 🔒 Security Notes

### Important:
1. **Never commit `.env` file** - Add to `.gitignore`
2. **Use secure passwords** for email accounts
3. **Limit dashboard access** in production:
   ```bash
   streamlit run dashboard.py --server.address 127.0.0.1
   ```
4. **Use HTTPS** for IP cameras in production
5. **Rotate API tokens** regularly
6. **Implement authentication** for web dashboard

### Privacy Considerations:
- ✅ This system is designed for safety, not surveillance
- ✅ Images/videos are deleted after processing (configurable)
- ✅ Use for authorized locations only
- ✅ Ensure compliance with local privacy laws
- ✅ Post signage indicating monitoring

---

## 📚 Advanced Features

### Custom Alert Rules
```python
# alert_manager.py - Modify check_alert_condition()
# Add custom logic based on time, day, location, etc.
```

### Database Integration
```python
# Store alerts in SQL database instead of JSON
# Example: PostgreSQL, SQLite, MongoDB
```

### Mobile App Integration
```python
# Create REST API using Flask
# Integrate with mobile app for real-time alerts
```

### Machine Learning Improvements
```python
# Train custom gender model on your data
# Implement face recognition for threat assessment
# Use action recognition to detect suspicious behavior
```

---

## 📈 Scalability Recommendations

### For 10+ Cameras:
1. Use GPU processing
2. Implement load balancing
3. Use distributed task queue (Celery)
4. Store data in database instead of JSON
5. Use message queue (RabbitMQ, Kafka)
6. Implement caching layer (Redis)

### For 100+ Cameras:
1. Microservices architecture
2. Kubernetes orchestration
3. Distributed database (Cassandra, DynamoDB)
4. CDN for video storage
5. Real-time analytics dashboard

---

## 📝 License

Educational Project | MIT License

---

## ⚠️ Educational Notice

This project is created for **educational purposes only**. It demonstrates:
- Computer Vision applications
- Multi-threading in Python
- API integration (Telegram, Email)
- Real-time data processing
- Web dashboard development

**Use responsibly and in compliance with local laws.**

---

## 🆘 Support & Contribution

For issues, suggestions, or improvements:
1. Check troubleshooting section
2. Review configuration carefully
3. Check model file paths
4. Verify API credentials

---

**Made with ❤️ for Women Safety and Educational Learning**
